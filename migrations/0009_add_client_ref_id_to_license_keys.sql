ALTER TABLE public.license_keys
ADD COLUMN client_reference_id TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_license_keys_client_reference_id
ON public.license_keys (client_reference_id);
