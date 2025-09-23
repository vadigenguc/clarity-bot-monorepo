-- Migration: 0004_add_channel_id_to_embeddings_and_channels_table.sql

-- Add channel_id column to document_embeddings table
ALTER TABLE public.document_embeddings
ADD COLUMN channel_id TEXT;

-- Update existing RLS policies for document_embeddings to include channel_id
-- First, drop the old policies
DROP POLICY IF EXISTS "Allow select for own workspace embeddings" ON public.document_embeddings;
DROP POLICY IF EXISTS "Allow insert for own workspace embeddings" ON public.document_embeddings;

-- Then, create new policies that also check channel_id
CREATE POLICY "Allow select for own workspace and channel embeddings" ON public.document_embeddings
FOR SELECT USING (
  (workspace_id = current_setting('request.headers.x-workspace-id', true))
  AND
  (channel_id = current_setting('request.headers.x-channel-id', true))
);

CREATE POLICY "Allow insert for own workspace and channel embeddings" ON public.document_embeddings
FOR INSERT WITH CHECK (
  (workspace_id = current_setting('request.headers.x-workspace-id', true))
  AND
  (channel_id = current_setting('request.headers.x-channel-id', true))
);

-- Create workspace_channels table
CREATE TABLE public.workspace_channels (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id TEXT NOT NULL,
  channel_id TEXT NOT NULL UNIQUE, -- Each channel can only be one project per workspace
  channel_name TEXT,
  is_enabled BOOLEAN DEFAULT TRUE,
  project_name TEXT, -- Optional: to name the project associated with the channel
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (workspace_id, channel_id) -- Ensure unique channel per workspace
);

-- Add RLS for workspace_channels table
ALTER TABLE public.workspace_channels ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow select for own workspace channels" ON public.workspace_channels
FOR SELECT USING (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

CREATE POLICY "Allow insert for own workspace channels" ON public.workspace_channels
FOR INSERT WITH CHECK (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

CREATE POLICY "Allow update for own workspace channels" ON public.workspace_channels
FOR UPDATE USING (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
) WITH CHECK (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

CREATE POLICY "Allow delete for own workspace channels" ON public.workspace_channels
FOR DELETE USING (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

-- Create workspace_subscriptions table
CREATE TABLE public.workspace_subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id TEXT NOT NULL UNIQUE,
  plan_id TEXT NOT NULL DEFAULT 'free', -- e.g., 'free', 'pro'
  max_channels_allowed INTEGER DEFAULT 1,
  current_channels_count INTEGER DEFAULT 0,
  stripe_customer_id TEXT, -- For billing integration
  stripe_subscription_id TEXT, -- For billing integration
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add RLS for workspace_subscriptions table
ALTER TABLE public.workspace_subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow select for own workspace subscriptions" ON public.workspace_subscriptions
FOR SELECT USING (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

-- Only allow service role to insert/update subscriptions (via backend logic)
-- No public INSERT/UPDATE policies for security
