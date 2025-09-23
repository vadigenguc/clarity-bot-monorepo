// This is a Netlify Function that triggers on a Polar webhook.
// It adds a new founding member to the Supabase database.

const { createClient } = require('@supabase/supabase-js');
const { Polar } = require('@polar-sh/sdk');

exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  const polar = new Polar({ token: process.env.POLAR_API_KEY });
  const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  try {
    const webhook = polar.webhooks.construct(
      event.body,
      event.headers['polar-signature'],
      process.env.POLAR_WEBHOOK_SECRET
    );

    const { type, payload } = webhook;

    if (type === 'subscription.created') {
      const { customer, id: subscription_id } = payload;
      const { email } = customer;

      const { error } = await supabase
        .from('founding_members')
        .insert([
          { 
            email: email, 
            polar_customer_id: customer.id,
            polar_subscription_id: subscription_id 
          }
        ]);

      if (error) {
        throw new Error(`Supabase error: ${error.message}`);
      }
    }

    return {
      statusCode: 200,
      body: 'Webhook processed successfully.',
    };
  } catch (error) {
    console.error('Error processing webhook:', error);
    return {
      statusCode: 400,
      body: `Webhook Error: ${error.message}`,
    };
  }
};
