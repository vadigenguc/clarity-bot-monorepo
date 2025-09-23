const { createClient } = require('@supabase/supabase-js');
const sgMail = require('@sendgrid/mail');
const { v4: uuidv4 } = require('uuid');
const crypto = require('crypto');

// Environment variables
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;
const POLAR_WEBHOOK_SECRET = process.env.POLAR_WEBHOOK_SECRET;
const SENDGRID_API_KEY = process.env.SENDGRID_API_KEY;
const SENDER_EMAIL = process.env.SENDER_EMAIL || 'noreply@claritypm.xyz';

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Initialize SendGrid
sgMail.setApiKey(SENDGRID_API_KEY);

/**
 * Correctly verifies a webhook signature from Polar based on their header format.
 * @param {object} headers The incoming request headers from the Netlify event.
 * @param {string} body The raw request body from the Netlify event.
 * @param {string} secretBase64 The base64-encoded webhook secret from environment variables.
 */
function verifyPolarSignature(headers, body, secretBase64) {
  const timestamp = headers['webhook-timestamp'];
  const signatureHeader = headers['webhook-signature'];

  if (!timestamp || !signatureHeader) {
    throw new Error('Missing webhook-timestamp or webhook-signature header');
  }

  // signatureHeader looks like: "v1,<base64_signature>"
  const parts = signatureHeader.split(',');
  if (parts.length !== 2 || parts[0] !== 'v1') {
    throw new Error('Invalid signature header format');
  }

  const signature = parts[1];
  const signedContent = `${timestamp}.${body}`;

  // The secret from Polar's dashboard is the raw secret, not base64.
  // The Standard Webhooks spec requires the key to be base64 decoded before use.
  // However, the crypto library's HMAC function expects the raw secret.
  // Per the Standard Webhooks spec, the secret key must be base64 decoded before use.
  const key = Buffer.from(POLAR_WEBHOOK_SECRET, 'base64');

  // Compute expected signature (base64)
  const expectedSignature = crypto
    .createHmac('sha256', key)
    .update(signedContent)
    .digest('base64');

  const sigBuf = Buffer.from(signature, 'base64');
  const expBuf = Buffer.from(expectedSignature, 'base64');

  if (sigBuf.length !== expBuf.length || !crypto.timingSafeEqual(sigBuf, expBuf)) {
    console.error('Signature mismatch.');
    console.error(`Received: ${signature}`);
    console.error(`Expected: ${expectedSignature}`);
    throw new Error('Invalid signature');
  }

  return true;
}


exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  if (!POLAR_WEBHOOK_SECRET) {
    console.error('POLAR_WEBHOOK_SECRET environment variable not set.');
    return { statusCode: 500, body: 'Server configuration error: POLAR_WEBHOOK_SECRET missing.' };
  }

  try {
    // Use the new, correct verification function
    verifyPolarSignature(event.headers, event.body, POLAR_WEBHOOK_SECRET);
    
    console.log('✅ Verified Polar webhook signature.');

    const webhookEvent = JSON.parse(event.body);

    console.log(`Received Polar webhook of type: ${webhookEvent.type}`);

    if (webhookEvent.type === 'order.paid') {
      const { customer, product_name, client_reference_id } = webhookEvent.data;
      const customer_email = customer.email;
      const tier = product_name || 'Founder';

      if (!customer_email) {
        console.error('Customer email missing in order.paid webhook.');
        return { statusCode: 400, body: 'Customer email missing.' };
      }

      const licenseKey = `CLARITY-FOUNDER-${uuidv4()}`;
      console.log(`Generated license key ${licenseKey} for ${customer_email} (${tier}).`);

      // Store license key in Supabase
      const { error } = await supabase
        .from('license_keys')
        .insert([
          {
            license_key: licenseKey,
            customer_email: customer_email,
            tier: tier,
            is_redeemed: false,
            client_reference_id: client_reference_id,
          },
        ]);

      if (error) {
        console.error('Error storing license key in Supabase:', error);
        return { statusCode: 500, body: `Error storing license key: ${error.message}` };
      }

      console.log(`License key ${licenseKey} stored in Supabase for ${customer_email}.`);

      // Send license key via SendGrid email
      const msg = {
        to: customer_email,
        from: SENDER_EMAIL,
        subject: 'Welcome to Clarity Founders\' Circle! Your Exclusive License Key Inside',
        html: `
          <p>Dear Founder,</p>
          <p>Welcome to the Clarity Founders' Circle! We're thrilled to have you as one of our early supporters.</p>
          <p>Your unique license key for activating your exclusive membership is:</p>
          <h3 style="font-family: monospace; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">${licenseKey}</h3>
          <p>You will receive the instructions to activate your membership before Clarity's official launch.</p> 
          <p>You will also receive an invitation to join our private Founders' Slack community to connect with other early adopters and directly influence our product roadmap.</p>
          <p>Thank you for being a part of Clarity's journey!</p>
          <p>Best regards,<br>The Clarity Team<br><a href="https://www.claritypm.xyz">www.ClarityPM.xyz</a></p>
        `,
      };

      await sgMail.send(msg);
      console.log(`License key email sent to ${customer_email}.`);

      return { statusCode: 200, body: 'Webhook processed successfully.' };
    } else {
      return { statusCode: 200, body: `Unhandled webhook type: ${webhookEvent.type}` };
    }
  } catch (error) {
    console.error('Error processing Polar webhook:', error);
    return { statusCode: 400, body: `Webhook Error: ${error.message}` };
  }
};
