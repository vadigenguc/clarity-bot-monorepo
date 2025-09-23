CREATE TABLE slack_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slack_message_ts TEXT NOT NULL UNIQUE,
  channel_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  message_text TEXT,
  raw_message_data JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE slack_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  slack_file_id TEXT NOT NULL UNIQUE,
  channel_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  file_name TEXT,
  file_type TEXT,
  file_size INTEGER,
  file_url TEXT,
  raw_file_data JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
