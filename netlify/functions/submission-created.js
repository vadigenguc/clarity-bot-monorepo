// This is a Netlify Function that triggers on form submission.
// It sends an autoresponder email using SendGrid.

// You need to install this dependency in your project: npm install @sendgrid/mail
const sgMail = require('@sendgrid/mail');

// IMPORTANT: Set your SendGrid API Key in the Netlify UI
// Go to Site settings > Build & deploy > Environment > Environment variables
sgMail.setApiKey(process.env.SENDGRID_API_KEY);

exports.handler = async (event) => {
  // We only care about POST requests from our form
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  try {
    const payload = JSON.parse(event.body).payload;
    const email = payload.data.email;

    // --- Email Content ---
    const msg = {
      to: email,
      from: 'clarity@clarity-bot.com', // IMPORTANT: Use an email you have verified with SendGrid
      subject: 'Thanks for joining the Clarity waitlist!',
      html: `
        <div style="font-family: sans-serif; line-height: 1.6; color: #333;">
          <h2>Welcome aboard!</h2>
          <p>Hi there,</p>
          <p>Thank you for signing up for the waitlist for <strong>Clarity Bot</strong>. We're excited to have you with us!</p>
          <p>We're working hard to build the best AI project assistant for Slack, and we'll be in touch with updates and an invitation for early access soon.</p>
          <p>Thank you for signing up for the waitlist for <strong>Clarity Bot</strong>. We're excited to have you with us!</p>
          <p>Meanwhile, you are most welcome to join as a <a href="https://buy.polar.sh/polar_cl_KMJ0F3LRgrqWRbPRLgPBXHbl5kWM1zXyyzYLX3gjXRN">Founding Member</a> and get the lifelong advantages.</p> 
          <p>You can reply to this mail for your inquiries.</p> 
          <p>Best,<br>The Clarity Bot Team<br><a href="https://www.clarit-bot.com">www.clarity-bot.com</a></p>
        </div>
      `,
    };

    await sgMail.send(msg);

    return {
      statusCode: 200,
      body: 'Email sent successfully.',
    };
  } catch (error) {
    console.error('Error sending email:', error);
    if (error.response) {
      console.error(error.response.body);
    }
    return {
      statusCode: 500,
      body: `Error: ${error.toString()}`,
    };
  }
};
