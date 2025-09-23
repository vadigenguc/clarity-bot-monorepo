CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id TEXT NOT NULL,
  source_type TEXT NOT NULL, -- e.g., 'message', 'file'
  source_id TEXT NOT NULL, -- e.g., slack_message_ts or slack_file_id
  content TEXT NOT NULL,
  embedding VECTOR(768), -- Assuming a common embedding dimension like 768 for sentence-transformers
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add RLS for the new table
ALTER TABLE public.document_embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow select for own workspace embeddings" ON public.document_embeddings
FOR SELECT USING (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);

CREATE POLICY "Allow insert for own workspace embeddings" ON public.document_embeddings
FOR INSERT WITH CHECK (
  workspace_id = current_setting('request.headers.x-workspace-id', true)
);
