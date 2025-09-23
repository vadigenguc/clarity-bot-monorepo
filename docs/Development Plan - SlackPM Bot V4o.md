# **Development Plan: Intelligent Jira Integration Slack Bot V4**

## **1\. Executive Summary**

This document outlines the development plan for a Slack bot designed to streamline project management workflows. The bot will dynamically learn project context from channel conversations and uploaded source documents, transcribe meetings, extract actionable tasks, and automate the creation of Jira tickets. A core feature is a robust authorization layer, ensuring only approved users can interact with the bot. A new conversational AI capability will allow authorized users to chat with the bot to ask questions about project status, technical implementation details, and past decisions. This plan is divided into eight distinct phases, allowing for iterative development, testing, deployment, and eventual commercialization on the Slack App Marketplace.

## **2\. Project Goals & Scope**

**Primary Goal:** To reduce manual effort, improve task tracking, and create an intelligent, conversational knowledge base for the project, while maintaining strict access control and privacy.

**Key Features:**

*   **Role-Based Access Control:** Only users explicitly authorized by a workspace admin can interact with the bot.
*   **Secure Interaction Model:** The bot will passively process messages and files from all public and private channels it's invited to (for contextual awareness). It will only respond or interact in Direct Messages (DMs), private group DMs (where all members are authorized), or public/private channels if the interaction originates from an authorized user and the channel is explicitly enabled for the bot. This ensures project-specific knowledge bases and interactions.
*   **Contextual Awareness:** Passively learn from all channel messages to understand project specifics, terminology, and stakeholders.
*   **Document Analysis:** Extract text and data from uploaded documents (PDF, DOCX, TXT, XLSX, CSV, etc.) to build a comprehensive knowledge base.
*   **Conversational Q&A:** Ask the bot questions about project status, task details, blockers, and past decisions (e.g., "What's the status of the newsletter task?", "How is the subscription feature implemented?").
*   **Meeting Transcription:** Transcribe externally uploaded video/audio recordings.
*   **Automated Summarization:** Generate concise summaries of meeting transcripts.
*   **Action Item Extraction:** Identify and parse actionable tasks from conversations and summaries.
*   **Jira Ticket Drafting:** Automatically create draft Jira tickets with pre-filled fields.
*   **In-Slack Approval Workflow:** Present drafted tickets for user review, editing, and approval directly within Slack DMs.
*   **Automated Jira Submission:** Push approved tickets to the designated Jira project and provide a confirmation link back in Slack.

## **3\. Phased Development Plan**

**Overall Development Roadmap:**
Phase 1 has been largely completed, establishing the foundational backend, authorization, multi-tenancy, and initial context engine. The subsequent phases (Phase 2 through Phase 6, and remaining tasks in Phase 7 and 8) represent the planned roadmap for adding core intelligent features, commercialization, and continuous improvement.

### **Phase 1: Foundational Setup, Authorization & Context**

*   **Objective:** Establish the core bot infrastructure, implement a robust user authorization system, and begin building its contextual knowledge base using a vectorization pipeline.
*   **Timeline:** 4 Weeks

**1.1 Slack App Setup**
[x] Register App on Slack API dashboard.
[x] Configure permissions (files:read, chat:write, commands, users:read, groups:read, im:history, mpim:history).
[x] Implement OAuth 2.0 for secure workspace installation.
Implementation Summary: The backend has been transitioned to Python with FastAPI. The `slack_bolt` library is integrated, and a FastAPI endpoint `/slack/events` is set up to receive all Slack events. OAuth scopes, Redirect URLs, and Slash Command URLs have been configured in the Slack App settings for HTTP Event Subscriptions (Socket Mode disabled).

**1.2 User Authorization System**
[x] Design database schema for storing authorized user IDs per workspace.
[x] Create admin-only slash commands: /bot-grant-access @user, /bot-revoke-access @user, /bot-list-authorized.
[x] Implement logic to verify that command issuer is a workspace Admin/Owner.
Implementation Summary: The `authorized_users` table schema is defined in `migrations/0001_create_authorized_users_table.sql` and includes `workspace_id` for multi-tenancy. The core logic for admin-only slash commands (`/bot-grant-access`, `/bot-revoke-access`, `/bot-list-authorized`) was part of the initial Node.js implementation and is conceptually covered. The `is_authorized` middleware now handles authorization checks.

**1.3 Secure Interaction Middleware**
[x] Develop a permission-checking function that triggers on every incoming event.
[x] For group DMs, the function must fetch all members of the conversation and verify that *every* member is authorized.
[x] If unauthorized, the bot remains silent.
Implementation Summary: The `is_authorized` middleware in `main.py` has been implemented. It now supports "Selective Interaction (Authorized Users Only)" in public and private channels: it passively processes messages/files from all channels it's invited to, but only responds/interacts if the initiating user is authorized. It also checks if the channel is explicitly enabled in the `workspace_channels` table. For DMs and group DMs, it ensures all members are authorized.

**1.4 Channel & Document Monitoring**
[x] Develop a service to listen for message and file upload events in authorized DMs and group DMs.
[x] Set up a scalable database to store raw message and document data.
Implementation Summary: The `slack_messages` and `slack_files` tables are defined in `migrations/0002_create_slack_data_tables.sql`. The `handle_message` and `handle_file_shared` event listeners in `main.py` store raw message content and file metadata in these tables, including `workspace_id` and `channel_id` for multi-tenancy and channel-level isolation. The Supabase client is configured with `SUPABASE_SERVICE_ROLE_KEY` and sets `x-workspace-id` and `x-channel-id` custom headers for RLS enforcement.

**1.5 Context Engine & Vectorization Pipeline**
[x] Integrate libraries for parsing various file formats (PDF, DOCX, XLSX, TXT, CSV).
[x] Implement a data pipeline: Chunk text from messages and documents, convert chunks to vector embeddings, and store them in a vector database.
Implementation Summary: The `pgvector` extension has been enabled in Supabase. The `document_embeddings` table is defined in `migrations/0003_create_document_embeddings_table.sql` and has been updated in `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql` to include `channel_id`. RLS policies for `document_embeddings` now filter by both `workspace_id` and `channel_id` using custom headers. Document parsing libraries (`pdfminer.six`, `python-docx`, `openpyxl`, `pandas`) and vectorization libraries (`sentence-transformers`, `tiktoken`) are installed. The `all-MiniLM-L6-v2` embedding model is loaded. Helper functions (`chunk_text`, `extract_text_from_file`, `get_embedding`, `process_and_store_content`) are implemented in `main.py`. The `handle_message` and `handle_file_shared` listeners now use `process_and_store_content` to extract text, chunk, embed, and store content in the `document_embeddings` table.

### **Phase 2: Meeting Transcription & Summarization**

*   **Objective:** Integrate speech-to-text for media files and summarization capabilities, feeding the results into the vector knowledge base.
*   **Timeline:** 2 Weeks

**2.1 Transcription Service Integration**
[ ] Select and integrate a third-party speech-to-text API (e.g., OpenAI Whisper).
[ ] Handle API authentication and job management.

**2.2 Media & Document File Handling**
[ ] Use Slack's API to detect and download all user-uploaded files in authorized contexts.
[ ] Route files to the appropriate service (transcription or document parsing).

**2.3 Summarization Engine**
[ ] Utilize a Large Language Model (e.g., via Gemini API) to generate summaries from transcripts.
[ ] Develop prompts engineered to focus on key decisions and outcomes.

**2.4 Post-Summary & Vectorize**
[ ] Post the summary as a threaded reply to the original file upload within the private DM or group DM.
[ ] Feed the transcript and summary into the vectorization pipeline (from Phase 1).

### **Phase 3: Action Item Extraction & Jira Integration**

*   **Objective:** Identify tasks from all sources and connect to the Jira API to create drafts.
*   **Timeline:** 3 Weeks

**3.1 Action Item NLP Model**
[ ] Fine-tune an LLM to identify phrases indicating a task, deadline, or assignment from any text source.
[ ] Extract structured data: {Task, Assignee, Due Date}.

**3.2 Secure Jira Connection**
[ ] Implement OAuth for secure, per-user authentication with the Jira Cloud API.
[ ] Store user tokens securely.

**3.3 Dynamic Field Mapping**
[ ] Develop logic to map extracted info to Jira fields.
[ ] Use channel context (from messages and docs) to intelligently suggest Project, Issue Type, and Priority.

**3.4 Test Draft Generation**
[ ] Build the API call to create a Jira issue but keep it in a "draft" or "to-do" state initially.

### **Phase 4: In-Slack User Approval Workflow**

*   **Objective:** Build the interactive UI for users to approve tickets within Slack DMs.
*   **Timeline:** 2 Weeks

**4.1 Interactive Card Design**
[ ] Use Slack's Block Kit to design a visually clear card for ticket proposals.
[ ] Card must display all key Jira fields.

**4.2 Edit Functionality**
[ ] Implement modals that allow users to edit fields like Summary, Description, or Assignee.

**4.3 Approval/Discard Logic**
[ ] Handle button clicks for "Create Ticket," "Edit," and "Discard."
[ ] Manage the state of the interactive message (e.g., disable buttons after action).

### **Phase 5: Final Jira Ticket Creation & Deployment**

*   **Objective:** Finalize the workflow by pushing tickets to Jira and preparing for production.
*   **Timeline:** 2 Weeks

**5.1 Push to Jira on Approval**
[ ] Trigger the final Jira API create issue call when a user clicks "Create Ticket."

**5.2 Confirmation & Linkback**
[ ] On successful creation, retrieve the new issue URL.
[ ] Update the original Slack message with a "Ticket Created!" confirmation and a link.

**5.3 Error Handling**
[ ] Implement robust error handling for API failures (Jira or Slack).
[ ] Provide clear, user-friendly error messages.

**5.4 Deployment & Logging**
[ ] Deploy the application to a production-ready cloud environment (e.g., Vercel).
[ ] Configure structured logging and monitoring.

### **Phase 6: Conversational AI & Project Q&A**

*   **Objective:** Transform the bot into an interactive project assistant that can answer questions and provide status updates based on its comprehensive knowledge base.
*   **Timeline:** 4 Weeks

**6.1 Semantic Search & Retrieval**
[ ] Develop a retrieval function that takes a user's question, converts it to an embedding, and performs a similarity search against the vector database to find the most relevant text chunks (the "context").

**6.2 RAG Prompt Engineering**
[ ] Engineer prompts for the Gemini LLM that combine the user's question with the retrieved context (Retrieval-Augmented Generation).
[ ] Instruct the model to answer the question *only* using the provided context to ensure factual accuracy.

**6.3 Jira Status Integration**
[ ] Augment the retrieval logic to also query the Jira API for a ticket's current status if a question is task-related.
[ ] Combine the live Jira status with conversational context from Slack for a complete answer.

**6.4 Conversation Handling**
[ ] Implement logic to handle multi-turn conversations and follow-up questions in a DM thread.

### **Phase 7: Commercialization & Marketplace Readiness**

*   **Objective:** To transform the bot from a single-instance tool into a secure, multi-tenant SaaS application ready for submission to the Slack App Marketplace.
*   **Timeline:** 4 Weeks

**7.1 Implement Multi-Tenancy**
[x] Update database schema to include a `workspace_id` on all relevant tables.
[x] Implement and enforce strict Row Level Security (RLS) policies in Supabase to ensure data isolation between workspaces.
Implementation Summary: The `workspace_id` is included in `authorized_users`, `slack_messages`, `slack_files`, and `document_embeddings` tables. RLS policies have been implemented for all these tables, enforcing data isolation by `workspace_id` and `channel_id` (for `document_embeddings`). RLS policies were updated to use `x-workspace-id` and `x-channel-id` custom HTTP headers for backend-initiated Supabase calls, removing the need for application-signed JWTs. New `workspace_channels` and `workspace_subscriptions` tables have been created with RLS policies to support channel-level project isolation and subscription tiers. This involved the creation of `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql`.

**7.2 Subscription & Billing Integration**
[x] Create `workspace_subscriptions` table to store subscription details and create RLS policies.
Implementation Summary: The `workspace_subscriptions` table has been created in `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql` to store subscription details, including `plan_id`, `max_channels_allowed`, and `current_channels_count`. RLS policies are in place to secure this table, allowing only the service role to modify it.
[ ] Define subscription tiers (e.g., Free, Pro) with feature limits (e.g., monthly transcription limits).
[ ] Integrate with Slack's billing/payment APIs.
[ ] Implement logic to check a workspace's subscription status and enforce feature limits.
[ ] Implement `/bot-enable-channel [channel_name]` slash command to enable a channel for bot interaction and context storage, checking against subscription limits.
[ ] Implement `/bot-disable-channel [channel_name]` slash command to disable a channel, freeing up a subscription slot.

**7.3 Onboarding & User Experience**
[ ] Create a friendly welcome message on first install.
[ ] Develop a clear onboarding flow that guides admins on how to grant user permissions.

**7.4 Marketplace Submission Prep**
[ ] Create a public-facing landing page with a Privacy Policy, ToS, and support contact.
[ ] Prepare all required materials for Slack's security review process.

### **Phase 8: Continuous Improvement & Advanced Features**

*   **Objective:** Refine the bot's intelligence and add value-add features post-launch.
*   **Timeline:** Ongoing

**8.1 Feedback Mechanism**
[ ] Create slash commands (/bot-feedback) for users to report inaccuracies in Q&A or ticket generation.
[ ] Store feedback to guide model improvements.

**8.2 Self-Learning Context**
[ ] Use approved/edited tickets and feedback on Q&A as training data to improve the accuracy of the context engine and field suggestions.

**8.3 Multi-Channel Support**
[x] Refactor the context engine to maintain separate vector knowledge bases for each private group DM the bot is in.
Implementation Summary: The `document_embeddings` table now includes `channel_id`, and the `workspace_channels` table has been created to explicitly track enabled channels as projects. The `is_authorized` middleware in `main.py` checks if a channel is enabled before processing events, ensuring project-specific knowledge bases and interactions. This also involved the creation of `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql`.

## **4\. Tech Stack Recommendation**

*   **Backend:** **Python (with FastAPI)**. This is the recommended choice due to its superior ecosystem for AI, data processing, and machine learning, which are core components of this project.
*   **Hosting:** **Vercel**. Provides excellent CI/CD, scalability, and developer experience for serverless applications.
*   **Database & Storage:** **Supabase**. A powerful backend-as-a-service platform using PostgreSQL. It will handle the relational database, user authorization, document storage, and the vector database using the pgvector extension.
*   **NLP/AI:** **Google Gemini API (via Vertex AI)**. Recommended for its superior large context window and strong reasoning capabilities, critical for both summarization and the RAG-based Q&A feature.
*   **Transcription:** **OpenAI Whisper API**. Recommended for its high accuracy and robustness in handling diverse, real-world audio conditions.
*   **Document Parsing:** Libraries like pdfminer, python-docx, openpyxl, and pandas.
*   **CI/CD:** GitHub Actions (natively integrates well with Vercel).

## **5\. Risks & Mitigation**

*   **Risk:** Inaccurate or "hallucinated" answers from the Q&A feature.
    *   **Mitigation:** Strictly adhere to a Retrieval-Augmented Generation (RAG) architecture that forces the LLM to answer based only on retrieved project data. Implement a feedback mechanism for users to flag bad answers.
*   **Risk:** High cost associated with vectorizing large volumes of data and frequent LLM calls.
    *   **Mitigation:** Implement intelligent caching and data chunking strategies. Monitor API usage closely from the start. Choose efficient embedding models.
*   **Risk:** Inaccurate text extraction from complex documents (e.g., PDFs with tables).
    *   **Mitigation:** Start with simpler formats (TXT, DOCX) and progressively add support for more complex ones. Use advanced parsing libraries.
*   **Risk:** Data privacy and security of conversations and Jira credentials.
    *   **Mitigation:** Use OAuth 2.0 for all external connections. Encrypt sensitive data at rest and in transit. Be transparent with users about what data is stored.
*   **Risk:** Hitting Slack or Jira API rate limits.
    *   **Mitigation:** Implement intelligent queueing and exponential backoff for API requests. Optimize to use fewer API calls where possible.
