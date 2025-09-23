CREATE TABLE IF NOT EXISTS public.license_keys (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    license_key TEXT UNIQUE NOT NULL,
    customer_email TEXT NOT NULL,
    tier TEXT NOT NULL,
    is_redeemed BOOLEAN DEFAULT FALSE NOT NULL,
    redeemed_by_workspace_id TEXT NULL,
    redeemed_by_user_id TEXT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    redeemed_at TIMESTAMP WITH TIME ZONE NULL
);

ALTER TABLE public.license_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Enable service role to manage all license keys"
ON public.license_keys
FOR ALL
USING (true) WITH CHECK (true);
