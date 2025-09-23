const { createClient } = require('@supabase/supabase-js');

exports.handler = async (event, context) => {
  if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.error('FATAL: Supabase environment variables are not set.');
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Server configuration error: Database credentials are missing.' }),
    };
  }

  const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

  if (event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: 'Method Not Allowed' }),
    };
  }

  let clientReferenceId;
  try {
    const body = JSON.parse(event.body);
    clientReferenceId = body.client_reference_id;
  } catch (error) {
    return {
      statusCode: 400,
      body: JSON.stringify({ error: 'Invalid request body. Expecting { "client_reference_id": "..." }' }),
    };
  }

  if (!clientReferenceId) {
    return {
      statusCode: 400,
      body: JSON.stringify({ error: 'Missing client_reference_id' }),
    };
  }

  try {
    // Look up the license key using the client_reference_id
    const { data: licenseKey, error: keyError } = await supabase
      .from('license_keys')
      .select('license_key')
      .eq('client_reference_id', clientReferenceId)
      .single();

    if (keyError || !licenseKey) {
      console.error('Error finding license key:', keyError);
      // This could happen if the user hits the success page before the webhook has been processed.
      // We'll return a "pending" status or a generic error.
      return {
        statusCode: 404,
        body: JSON.stringify({ error: 'License key not found or still processing. Please check your email or contact support.' }),
      };
    }
      
    // Return the found key
    return {
      statusCode: 200,
      body: JSON.stringify({ license_key: licenseKey.license_key }),
    };

  } catch (error) {
    console.error('Error in get-license-key function:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Internal Server Error' }),
    };
  }
};
