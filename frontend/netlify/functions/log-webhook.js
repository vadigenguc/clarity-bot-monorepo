// functions/log-webhook.js
exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  console.log('🔔 Received webhook event');
  console.log('--- Headers ---');
  console.log(JSON.stringify(event.headers, null, 2));

  console.log('--- Body ---');
  console.log(event.body);

  return {
    statusCode: 200,
    body: 'Webhook received (logging only, no verification)',
  };
};
