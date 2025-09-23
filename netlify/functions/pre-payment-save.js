// This function saves user details from the pre-payment modal to Supabase.

const { createClient } = require('@supabase/supabase-js');

exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method Not Allowed' };
  }

  const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
  const { name, surname, email, phone, country } = JSON.parse(event.body);

  if (!name || !surname || !email) {
    return { statusCode: 400, body: 'Name, surname, and email are required.' };
  }

  try {
    // Upsert ensures that if the email already exists, we update the record.
    // If it doesn't exist, a new one is created.
    const { data, error } = await supabase
      .from('founding_members')
      .upsert(
        { 
          email, 
          name, 
          surname, 
          phone, 
          country,
          payment_status: 'initiated',
          updated_at: new Date().toISOString() 
        },
        { onConflict: 'email' }
      )
      .select()
      .single();

    if (error) {
      throw new Error(`Supabase error: ${error.message}`);
    }

    return {
      statusCode: 200,
      body: JSON.stringify({ id: data.id }),
    };
  } catch (error) {
    console.error('Error saving user data:', error);
    return {
      statusCode: 500,
      body: `Server Error: ${error.message}`,
    };
  }
};
