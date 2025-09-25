# **Development Plan: Intelligent Jira Integration Slack Bot V4**

## **1\. Executive Summary**

This document outlines the development plan for a Slack bot designed to streamline project management workflows. The bot will dynamically learn project context from channel conversations and uploaded source documents, transcribe meetings, extract actionable tasks, and automate the creation of Jira tickets. A core feature is a robust authorization layer, ensuring only approved users can interact with the bot. A new conversational AI capability will allow authorized users to chat with the bot to ask questions about project status, technical implementation details, and past decisions. This plan is divided into nine distinct phases, allowing for iterative development, testing, deployment, and eventual commercialization on the Slack App Marketplace.

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
Phase 1 has been largely completed, establishing the foundational backend, authorization, multi-tenancy, and initial context engine. The subsequent phases (Phase 2 through Phase 9) represent the planned roadmap for adding core intelligent features, commercialization, and continuous improvement.

### **Phase 1: Foundational Setup, Authorization & Context**

*   **Objective:** Establish the core bot infrastructure, implement a robust user authorization system, and begin building its contextual knowledge base using a vectorization pipeline.
*   **Timeline:** 4 Weeks

**1.1 Slack App Setup**
[x] Register App on Slack API dashboard.
[x] Configure permissions (files:read, chat:write, commands, users:read, groups:read, im:history, mpim:history).
[x] Implement OAuth 2.0 for secure workspace installation.
Implementation Summary: The backend has been transitioned to Python with FastAPI. The `slack_bolt` library is integrated, and a FastAPI endpoint `/slack/events` is set up to receive all Slack events. OAuth scopes, Redirect URLs, and Slash Command URLs have been configured in the Slack App settings for HTTP Event Subscriptions (Socket Mode disabled).
**Post-Implementation Refactoring & Bug Fixes:**
- **Dependency Management:** A `requirements.txt` file was created to manage and pin all Python dependencies. This was critical to resolve a series of cascading dependency conflicts between `sentence-transformers`, `transformers`, `huggingface-hub`, and `supabase`, which were causing critical application startup failures. The final stable versions were locked in to ensure a reproducible environment.
- **Asynchronous Handling:** The initial implementation used the synchronous `slack_bolt.App`, which caused a `RuntimeWarning: coroutine was never awaited` and prevented the bot from responding to messages. This was fixed by migrating to `slack_bolt.async_app.AsyncApp` and `AsyncSlackRequestHandler`, ensuring proper asynchronous event handling within the FastAPI framework.

**1.2 User Authorization System**
[x] Design database schema for storing authorized user IDs per workspace.
[x] Create admin-only slash commands: /bot-grant-access @user, /bot-revoke-access @user, /bot-list-authorized.
[x] Implement logic to verify that command issuer is a workspace Admin/Owner.
Implementation Summary: The `authorized_users` table schema is defined in `migrations/0001_create_authorized_users_table.sql` and includes `workspace_id` for multi-tenancy. The core logic for admin-only slash commands (`/bot-grant-access`, `/bot-revoke-access`, `/bot-list-authorized`) was part of the initial Node.js implementation and is conceptually covered. The `is_authorized` middleware now handles authorization checks.
**Post-Implementation Refactoring & Bug Fixes:**
- **Admin Command Implementation:** The `/bot-list-authorized` and `/bot-grant-access` commands were fully implemented in `main.py`. Additionally, the `/bot-revoke-access` command was proactively implemented to complete the user management feature set.
- **Admin Authorization Logic:** A critical flaw was fixed where workspace admins could not run admin-level commands unless they had also been explicitly added to the `authorized_users` list. A new `is_user_admin` helper function was created, which uses the `client.users_info` Slack API method to check a user's admin/owner status. This check now runs at the beginning of all admin commands, bypassing the standard authorization flow and allowing admins to manage the bot correctly.
- **Flexible User Input:** The grant/revoke commands were improved to handle user lookups flexibly. They now accept both the standard Slack user mention format (e.g., `<@U12345>`) and plain text usernames (e.g., `@vadi`). This was achieved by implementing a `find_user_id_by_name` helper function that paginates through the `client.users_list` API to find a matching user ID.
- **Database Schema Correction:** A bug was fixed where the `/bot-grant-access` command was attempting to insert a `granted_by` column that did not exist in the `authorized_users` table schema. The code was corrected to align with the schema defined in the migration file.

**1.3 Secure Interaction Middleware**
[x] Develop a permission-checking function that triggers on every incoming event.
[x] For group DMs, the function must fetch all members of the conversation and verify that *every* member is authorized.
[x] If unauthorized, the bot remains silent.
Implementation Summary: The `is_authorized` middleware in `main.py` has been implemented. It now supports "Selective Interaction (Authorized Users Only)" in public and private channels: it passively processes messages/files from all channels it's invited to, but only responds/internacts if the initiating user is authorized. It also checks if the channel is explicitly enabled in the `workspace_channels` table. For DMs and group DMs, it ensures all members are authorized.
**Post-Implementation Refactoring & Bug Fixes:**
- **Authorization Logic Refactoring:** The main `check_authorization` function was significantly refactored for clarity and maintainability. The monolithic function was broken down into smaller, single-purpose helper functions: `is_channel_enabled`, `is_user_authorized`, and `are_all_group_members_authorized`.
- **Context Parsing Bug:** A critical bug was fixed where the application would crash with an `AttributeError` when processing events. The issue was that the authorization logic was incorrectly attempting to parse context variables (`user_id`, `channel_id`) from the event `body` instead of the `context` object provided by `slack_bolt`. The logic was corrected to pull these IDs from the correct source.
- **Supabase Client Initialization:** A runtime crash was fixed related to the `supabase` library upgrade. The code was updated to use the new `ClientOptions` object when creating a Supabase client with custom headers, resolving an `AttributeError: 'dict' object has no attribute 'headers'`.

**1.4 Channel & Document Monitoring**
[x] Develop a service to listen for message and file upload events in authorized DMs and group DMs.
[x] Set up a scalable database to store raw message and document data.
[x] Provide a confirmation message to the user upon successful document ingestion.
Implementation Summary: The `slack_messages` and `slack_files` tables are defined in `migrations/0002_create_slack_data_tables.sql`. The `handle_message` and `handle_file_shared` event listeners in `main.py` store raw message content and file metadata in these tables, including `workspace_id` and `channel_id` for multi-tenancy and channel-level isolation. The Supabase client is configured with `SUPABASE_SERVICE_ROLE_KEY` and sets `x-workspace-id` and `x-channel-id` custom headers for RLS enforcement.
**Post-Implementation Refactoring & Bug Fixes:**
- **Asynchronous Database Calls:** A major performance bottleneck was resolved. The initial implementation used synchronous Supabase database calls (`.execute()`) inside `async` functions, which blocked the entire application's event loop. This was fixed by wrapping all database calls in `asyncio.to_thread`, ensuring that these blocking I/O operations run in a separate thread and do not freeze the main application. This fixed a `TypeError: object APIResponse can't be used in 'await' expression` and dramatically improved bot responsiveness.
- **Conditional Document Ingestion:** The bot now only ingests documents if explicitly tagged and instructed (e.g., via an `@Clarity Bot ingest` mention) in the accompanying message. This prevents ingestion of out-of-project-scope documents and reduces noise/garbage data. A confirmation message is provided upon successful, instructed ingestion.

**1.5 Context Engine & Vectorization Pipeline**
[x] Integrate libraries for parsing various file formats (PDF, DOCX, XLSX, TXT, CSV).
[x] Implement a data pipeline: Chunk text from messages and documents, convert chunks to vector embeddings, and store them in a vector database.
Implementation Summary: The `pgvector` extension has been enabled in Supabase. The `document_embeddings` table is defined in `migrations/0003_create_document_embeddings_table.sql` and has been updated in `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql` to include `channel_id`. RLS policies for `document_embeddings` now filter by both `workspace_id` and `channel_id` using custom headers. Document parsing libraries (`pdfminer.six`, `python-docx`, `openpyxl`, `pandas`) and vectorization libraries (`sentence-transformers`, `tiktoken`) are installed. The `all-MiniLM-L6-v2` embedding model is loaded. Helper functions (`chunk_text`, `extract_text_from_file`, `get_embedding`, `process_and_store_content`) are implemented in `main.py`. The `handle_message` and `handle_file_shared` listeners now use `process_and_store_content` to extract text, chunk, embed, and store content in the `document_embeddings` table.

### **Phase 2: Landing Page & Waitlist Implementation**

*   **Objective:** Develop and deploy a public-facing landing page with a waitlist feature, integrated with Supabase for email storage.
*   **Timeline:** 2 Weeks

**2.1 Landing Page Refactoring & Deployment**
[x] Refactor existing single HTML file (if any) into a modular structure with separate HTML, CSS, and JavaScript files.
[ ] Design a compelling landing page to attract early adopters.
[ ] Deploy the landing page to Netlify.

**2.2 Waitlist Feature Implementation**
[x] Configure Netlify Forms for the landing page.
[x] Implement a form on the landing page for users to submit their email addresses.
[x] Implement basic validation and error handling for email submissions.
[x] Provide a confirmation message to the user upon successful waitlist signup.
**Implementation Summary:**
- The waitlist form in `web/index.html` has been configured to use Netlify Forms by adding the `data-netlify="true"` attribute and a hidden `form-name` input.
- Client-side email validation and error display have been implemented in `web/script.js`.
- A toaster notification provides a success message without a page reload, also handled by `web/script.js`.
- A Netlify Function (`netlify/functions/submission-created.js`) has been created to send an autoresponder email via SendGrid upon successful submission.


### **Phase 3: Meeting Transcription & Summarization**

*   **Objective:** Integrate speech-to-text for media files and summarization capabilities, feeding the results into the vector knowledge base.
*   **Timeline:** 2 Weeks

**3.1 Transcription Service Integration**
[x] Select and integrate a third-party speech-to-text API (OpenAI Whisper).
[x] Handle API authentication and job management.
Implementation Summary: OpenAI Whisper API has been selected for speech-to-text transcription. For files under 25MB, direct transcription via FastAPI is used. For larger files, a job queuing system using Google Cloud Pub/Sub and a Google Cloud Run Job background worker is implemented. Docker images for both the FastAPI backend and the worker are built using Google Cloud Build.

**3.2 Media & Document File Handling**
[x] Use Slack's API to detect and download all user-uploaded files in authorized contexts.
[x] Route files to the appropriate service (transcription or document parsing).
Implementation Summary:
- **Asynchronous File Downloads:** A performance bottleneck in the `handle_file_shared` event listener was fixed. The synchronous `requests.get()` call, which blocked the entire event loop during file downloads, was replaced with an asynchronous equivalent using `httpx.AsyncClient`. This ensures the bot remains responsive to other events even when downloading large files.
- **Conditional Routing for Transcription:**
    - **Small Files (<25MB):** Audio/video files under 25MB are directly transcribed by the FastAPI application using the OpenAI Whisper API.
    - **Large Files (>25MB):** For audio/video files exceeding 25MB (Whisper API limit), a new asynchronous job queuing architecture is implemented using Google Cloud Pub/Sub and Google Cloud Run Jobs:
        1.  **Slack Webhook to Vercel (FastAPI):** When a user uploads a large audio/video file and explicitly instructs the bot to ingest it, the FastAPI endpoint receives the Slack webhook.
        2.  **Instant Job Queuing (FastAPI side):** The Vercel function (FastAPI) does *not* download or process the file. Instead, it quickly creates a "job" (e.g., `{ "file_url": "...", "workspace_id": "...", "channel_id": "...", "file_id": "...", "file_name": "...", "user_id": "..." }`) and pushes it into a Google Cloud Pub/Sub topic. It then immediately sends a 200 OK response back to Slack to confirm receipt, ensuring responsiveness.
        3.  **Background Worker (Google Cloud Run Job):** A separate service, a Background Worker deployed as a Google Cloud Run Job, is responsible for constantly monitoring and processing jobs from the Google Cloud Pub/Sub queue.
        4.  **Processing in the Worker:** When the Cloud Run Job picks up a new job, it performs the heavy lifting inside its dedicated, long-timeout environment:
            *   Downloads the large file from Slack.
            *   Uses `pydub` to split the file into smaller chunks (e.g., five 20MB chunks for a 100MB file).
            *   Sends all chunks to the Whisper API in parallel for transcription.
            *   Receives the individual text transcripts and stitches them together.
            *   Saves the final, complete transcript to the Supabase database (into `document_embeddings` table with `source_type='transcription'`).
            *   Sends a final Slack message to the user (e.g., "Your transcription for `[file_name]` is complete!") to confirm completion.
- **Docker Image Creation & Artifact Registry (via Google Cloud Build):**
    - Docker images for both the FastAPI backend and the worker are built using Google Cloud Build, eliminating the need for local Docker Desktop installation on Windows Home editions.
    - **Artifact Registry:** Docker images are stored in Google Cloud Artifact Registry (e.g., `europe-west1-docker.pkg.dev/clarity-b0t/clarity-worker-repo/clarity-worker:latest`). A dedicated Docker repository (`clarity-worker-repo`) is created in Artifact Registry for this purpose.
    - **Cloud Build Triggers:** Automated Cloud Build triggers are configured for both the backend and worker. These triggers monitor GitHub for pushes to the `main` branch, build the respective Dockerfiles, and push the tagged images to Artifact Registry. The worker's Cloud Build uses a `cloudbuild.yaml` to explicitly define build and push steps, and sets logging to `CLOUD_LOGGING_ONLY`.
    - **Worker Dockerfile:** The `worker/Dockerfile` now uses `python:3.10-slim-bullseye` as its base image and includes installation of `ffmpeg`, `libsm6`, and `libxext6` for audio processing.
    - **Worker Requirements:** The `worker/requirements.txt` includes `google-cloud-pubsub`, `openai`, `pydub`, `python-dotenv`, and `supabase`.
    - **Cloud Run Job Deployment:** The `clarity-transcription-worker` Cloud Run Job has been successfully deployed, configured to pull its image from Artifact Registry and with all necessary environment variables (Supabase URL/Key, OpenAI API Key, GCP Project ID, Pub/Sub Topic/Subscription names) correctly set.
    - **Cloud Scheduler:** A Cloud Scheduler job (`clarity-worker-scheduler`) has been successfully created and enabled to publish a message to the `clarity-transcription-jobs` Pub/Sub topic every 5 minutes, which triggers the Cloud Run Job.
- **File Size Limit Update:** The `MAX_DIRECT_TRANSCRIPTION_SIZE_MB` in `backend/main.py` has been updated from 25MB to 20MB.
- **GCP Project ID:** `clarity-b0t` is the Google Cloud Project ID used for all GCP resources.
- **GCP Region:** `europe-west1` is the Google Cloud region used for all regional GCP resources to minimize latency and data transfer costs.

**3.3 Summarization Engine**
*   **Objective:** Utilize a Large Language Model (e.g., via Gemini API) to generate summaries from transcripts, focusing on key decisions and outcomes, and feed the results into the vector knowledge base.
*   **Timeline:** 2 Weeks (Estimated)
*   **Action Details:**
    *   [x] **Implement Summarization Logic:** Developed a new asynchronous function, `summarize_text`, within `backend/services/llm_service.py`. This function uses the `LLMServiceManager.generate_text` method with the `gemini-pro` model to generate a concise summary from provided text and a summarization prompt.
    *   [x] **Integrate Summarization into Worker:** Modified `worker/worker.py`. After a large media file has been transcribed by the Whisper API, the `worker.py` script now calls the `summarize_text` function with the full transcript and the loaded summarization prompt. The worker then receives the generated summary.
    *   [x] **Post Summary to Slack:** Implemented logic in `worker/worker.py` to post the generated summary as a threaded reply to the original Slack message where the file was uploaded, providing immediate user feedback.
    *   [x] **Vectorize Summary:** Ensured that the generated summary is also fed into the existing vectorization pipeline (via `process_and_store_content`) to update the project's knowledge base with the summarized content.
*   **Testing:**
    *   [x] **Unit Tests:** Added unit tests for the new summarization logic in `tests/test_llm_service.py`. These tests mock the `LLMServiceManager.generate_text` call and verify the `summarize_text` function's behavior, including prompt usage and error handling.
    *   [x] **Integration Tests:** Extended `tests/test_integration.py` to include an end-to-end test case (`test_process_transcription_job_with_summarization_integration`) that simulates the entire flow: file transcription, summarization, Slack posting, and vectorization.
    *   [x] **Execute Tests:** All tests were run using `pytest tests/` and passed successfully.

**Implementation Summary:**
The `summarize_text` asynchronous method was added to the `LLMServiceManager` class in `backend/services/llm_service.py`, utilizing the `gemini-pro` model. The `process_transcription_job` function in `worker/worker.py` was updated to load the `summarization_prompt`, call `llm_service_manager.summarize_text`, store the summary in Supabase, and send it as a threaded reply to Slack.
- __Summarization Logic in `backend/services/llm_service.py`__: A new asynchronous method `summarize_text` was added to the `LLMServiceManager` class. This method utilizes the `gemini-pro` model via `generate_text` to create concise summaries from provided text and a summarization prompt.

- __Integration into `worker/worker.py`__: The `process_transcription_job` function in the worker was updated to:

  - Load the `summarization_prompt` using `backend.utils.prompt_loader.load_prompt`.
  - Call `llm_service_manager.summarize_text` with the full transcript to generate a summary.
  - Store the generated summary in the Supabase `document_embeddings` table with `source_type='summary'`.
  - Send the summary as a threaded reply to the original Slack message, providing immediate user feedback.


**Testing Summary:**
Unit tests for `summarize_text` were added to `tests/test_llm_service.py`, mocking `generate_text` to verify functionality. An integration test (`test_process_transcription_job_with_summarization_integration`) was added to `tests/test_integration.py` to simulate the full worker flow, including transcription, summarization, Supabase storage, and Slack messaging, with extensive mocking of external dependencies. All 20 tests passed successfully.
1. __Unit Tests for `summarize_text` (`tests/test_llm_service.py`)__: New unit tests were added to verify the correct behavior of the `summarize_text` method, including successful summary generation and proper handling of failure scenarios (e.g., when `generate_text` returns `None`). These tests mocked the underlying `generate_text` call to isolate the `summarize_text` logic.
2. __Integration Test for Summarization Flow (`tests/test_integration.py`)__: A new integration test `test_process_transcription_job_with_summarization_integration` was added. This test simulates the entire worker flow, from downloading a file and transcribing it, to generating a summary, storing both in Supabase, and sending Slack messages. Extensive mocking was used for external dependencies (Slack API, Supabase, LLM services) to ensure the integration logic was correctly exercised. The use of `unittest.mock.ANY` was crucial for correctly asserting calls to `process_and_store_content` with dynamically created Supabase client instances.

All 20 tests passed successfully, confirming the robust implementation and integration of the summarization engine.


**3.4 Post-Summary & Vectorize**
[x] Post the summary as a threaded reply to the original file upload within the private DM or group DM.
[x] Feed the transcript and summary into the vectorization pipeline (from Phase 1).

### **Phase 3.5: Advanced Content Ingestion & Relevance**

*   **Objective:** Implement intelligent filtering for message content and flexible, delayed processing for attachments based on user instructions.
*   **Timeline:** 3 Weeks

**3.5.1 Implement LLM-based Relevance Filtering for Messages**
*   **Objective:** Ensure only project-relevant text conversations are ingested into the knowledge base.
*   **Action Details:**
    *   [ ] Define Relevance Criteria: Collaborate with project stakeholders to establish clear criteria and examples of what constitutes "project-related" content.
    *   [ ] Integrate Lightweight LLM for Classification: Add `google-cloud-aiplatform` to `backend/requirements.txt` for Gemini API access. Initialize a Google Gemini API client (via Vertex AI) in `backend/main.py` or a new `llm_service.py` module. Develop a function (e.g., `is_message_relevant(text: str) -> bool`) that uses Gemini to classify message text.
    *   [ ] Develop Prompt for Relevance Classification: Craft a concise and effective prompt for the LLM to determine if a given message is relevant to the project's context.
    *   [ ] Modify `handle_message` for Conditional Ingestion: In `backend/main.py`, update the `handle_message` function to call `is_message_relevant` before calling `process_and_store_content`. If `is_message_relevant` returns `False`, log the decision and skip ingestion. (Optional) Provide a subtle, private notification to the user if a message was ignored due to irrelevance, with an option to override.

**3.5.2 Enhance Attachment Ingestion Instructions**
*   **Objective:** Broaden the range of triggering words and improve detection of instructions within threads or quoted messages for file processing.
*   **Action Details:**
    *   [ ] Expand Triggering Keyword List: Define a configurable list of keywords (e.g., "ingest", "read", "process", "analyze", "review") in environment variables or a configuration file. Update the logic in `handle_file_shared` to check for any of these keywords in the associated message.
    *   [ ] Implement Robust Thread/Quote Instruction Detection: For `file_shared` events: If the initial message doesn't contain the instruction, enhance `handle_file_shared` to check if the `file_shared` event is part of a thread. If `event.get('thread_ts')` exists, fetch the thread's history and search for the instruction keywords within messages in that thread. For `message` events (referencing old files): Modify `handle_message` to detect if a message contains a bot mention, an instruction keyword, *and* a reference to a Slack file (e.g., a quoted block containing a file, or a direct Slack file link). Develop a helper function (e.g., `extract_file_id_from_message(message_text: str) -> str | None`) to parse the message and extract a Slack `file_id`.

**3.5.3 Enable Delayed File Processing via Message Events**
*   **Objective:** Allow users to instruct the bot to process previously uploaded files (even with expired temporary URLs) by referencing them in new messages.
*   **Action Details:**
    *   [ ] Create a `process_slack_file_by_id` Function: Extract the core logic from `handle_file_shared` that fetches file info, downloads, transcribes/extracts text, and stores content. Create a new asynchronous function, `process_slack_file_by_id(file_id: str, workspace_id: str, channel_id: str, user_id: str, rls_supabase_client: Client, say_function)` that takes a `file_id` and other necessary context, and performs the file processing. This function will always use `slack_app.client.files_info(file=file_id)` to get the latest valid URL.
    *   [ ] Integrate into `handle_message`: If `handle_message` detects a bot mention, an instruction keyword, and a valid `file_id` (extracted from the message as per 3.5.2.2), it should call `process_slack_file_by_id` with the extracted `file_id`. Provide appropriate user feedback (e.g., "Processing your requested file...") and error handling.

### **Phase 3.6: Core LLM & Prompt Management Modularity**

*   **Objective:** Refactor LLM interactions and prompt definitions into a modular, extensible structure to support current and future AI features.
*   **Timeline:** 1 Week (Estimated)

**3.6.1 Create `backend/services/llm_service.py`**
*   **Action Details:**
    *   [x] Create the `backend/services` directory.
    *   [x] Create `backend/services/llm_service.py`.
    *   [x] Move LLM client initialization (e.g., `openai_client`, future Gemini client) from `main.py` to `llm_service.py`.
    *   [x] Implement a class (e.g., `LLMServiceManager`) to manage different LLM providers (OpenAI, Gemini).
    *   [x] Add generic asynchronous methods for LLM interaction (e.g., `LLMServiceManager.generate_text(model_name, prompt, ...)`).
    *   [x] Update `backend/requirements.txt` to include `google-cloud-aiplatform` and any other necessary LLM client libraries for Gemini.
    **Implementation Summary:** The `backend/services` directory and `llm_service.py` were created. The `LLMServiceManager` class was implemented to encapsulate OpenAI and Google Gemini client initialization and provide a unified `generate_text` method. `backend/requirements.txt` was updated to include `google-cloud-aiplatform`.

**3.6.2 Create `backend/prompts/` directory and Prompt Loader**
*   **Action Details:**
    *   [x] Create the `backend/prompts` directory.
    *   [x] Create `backend/prompts/relevance_prompt.json` (for message relevance classification).
    *   [x] Create `backend/prompts/summarization_prompt.json` (for meeting summarization).
    *   [x] Create `backend/prompts/qna_prompt.json` (for conversational Q&A).
    *   [x] Create `backend/prompts/jira_ticket_prompt.json` (for Jira ticket generation).
    *   [x] Implement a utility function (e.g., `load_prompt(prompt_name: str) -> str`) in a new `backend/utils/prompt_loader.py` module to load prompts from these JSON files.
    **Implementation Summary:** The `backend/prompts` directory was created, along with JSON files for `relevance_prompt`, `summarization_prompt`, `qna_prompt`, and `jira_ticket_prompt`. The `backend/utils` directory and `prompt_loader.py` were created, implementing a `load_prompt` function to dynamically load these prompts.

**3.6.3 Refactor `backend/main.py` and `worker/worker.py`**
*   **Action Details:**
    *   [x] Update imports in `backend/main.py` and `worker/worker.py` to use the new `llm_service` and `prompt_loader` modules.
    *   [x] Replace hardcoded prompts with calls to the `load_prompt` utility.
    *   [x] Replace direct `openai_client` calls with calls to the `LLMServiceManager`.
    **Implementation Summary:** `backend/main.py` and `worker/worker.py` were refactored. Direct `openai_client` imports and initializations were removed. Imports for `llm_service_manager` and `load_prompt` were added. The `transcribe_audio` function in `main.py` and `transcribe_audio_chunk` in `worker.py` were updated to use `llm_service_manager.generate_text` for transcription.

**3.6.4 Testing**
*   **Objective:** Verify the functionality and integration of the LLM service manager and prompt loader.
*   **Action Details:**
    *   [x] **Unit Tests for `llm_service.py`:**
        *   [x] Test `LLMServiceManager` initialization with and without API keys.
        *   [x] Mock LLM API calls to test `generate_text` for OpenAI and Gemini models.
        *   [x] Test error handling for unsupported models and API failures.
    *   [x] **Unit Tests for `prompt_loader.py`:**
        *   [x] Test successful loading of existing prompts.
        *   [x] Test error handling for `FileNotFoundError` (missing JSON file).
        *   [x] Test error handling for `KeyError` (missing 'prompt' key in JSON).
        *   [x] Test error handling for `json.JSONDecodeError` (malformed JSON).
    *   [x] **Integration Tests for `backend/main.py` and `worker/worker.py`:**
        *   [x] Verify that `main.py` correctly imports and uses `llm_service_manager` and `load_prompt` for transcription.
        *   [x] Verify that `worker/worker.py` correctly imports and uses `llm_service_manager` and `load_prompt` for transcription.
        *   [ ] (Future) Extend integration tests to cover summarization and relevance filtering once those features are implemented.
    **Testing Summary:** A comprehensive test suite was developed and executed to validate the new modular components:
    1.  **`tests/` directory**: Created to house all test files.
    2.  **`tests/conftest.py`**: Configured to mock external dependencies such as `AsyncOpenAI`, `GenerativeModel`, `aiplatform.init`, `pubsub_v1.PublisherClient`, `pubsub_v1.SubscriberClient`, and `google.auth.default`. This ensured isolated and reliable testing of the LLM and Pub/Sub integrations.
    3.  **`tests/test_llm_service.py`**: Unit tests were implemented to verify the correct initialization of OpenAI and Gemini clients, and the functionality of the `generate_text` method for various models (OpenAI GPT-4, OpenAI Whisper-1, Gemini-Pro), including success and failure scenarios.
    4.  **`tests/test_prompt_loader.py`**: Unit tests were implemented to confirm the successful loading of prompts from JSON files, and to handle cases of missing files, missing keys, and malformed JSON.
    5.  **`tests/test_integration.py`**: Integration tests were implemented to ensure that `transcribe_audio` in `backend/main.py` and `transcribe_audio_chunk` in `worker/worker.py` correctly utilize the `LLMServiceManager` for transcription.
    All 17 tests passed successfully.
	
**3.6: Core LLM & Prompt Management Modularity - Implementation and Testing Report**

**Implementation Summary:**
The Core LLM & Prompt Management Modularity (Phase 3.6) has been successfully implemented. This involved:
1.  **`backend/services/llm_service.py`**: Created to encapsulate LLM client initialization and text generation logic for both OpenAI (GPT-4, Whisper) and Google Gemini (gemini-pro). It handles API key/project ID retrieval from environment variables and provides a unified `generate_text` interface.
2.  **`backend/prompts/` directory and Prompt Loader**: A new `prompts` directory was created to store LLM prompts in JSON files (`relevance_prompt.json`, `summarization_prompt.json`, `qna_prompt.json`, `jira_ticket_prompt.json`). The `backend/utils/prompt_loader.py` utility was developed to dynamically load these prompts, ensuring modularity and easy management.
3.  **Refactoring `backend/main.py` and `worker/worker.py`**: Both `main.py` (FastAPI application) and `worker.py` (Cloud Run Job worker) were refactored to integrate the new `LLMServiceManager` and `prompt_loader` utility. This significantly reduced coupling and improved code organization.
4.  **Lazy Pub/Sub Client Initialization**: Pub/Sub client initialization in both `backend/main.py` and `worker/worker.py` was made lazy to prevent `DefaultCredentialsError` during module import, especially in test environments or when specific GCP credentials are not immediately available.

**Testing Summary:**
A comprehensive test suite was developed and executed to validate the new modular components:
1.  **`tests/` directory**: Created to house all test files.
2.  **`tests/conftest.py`**: Configured to mock external dependencies such as `AsyncOpenAI`, `GenerativeModel`, `aiplatform.init`, `pubsub_v1.PublisherClient`, `pubsub_v1.SubscriberClient`, and `google.auth.default`. This ensured isolated and reliable testing of the LLM and Pub/Sub integrations.
3.  **`tests/test_llm_service.py`**: Unit tests were implemented to verify the correct initialization of OpenAI and Gemini clients, and the functionality of the `generate_text` method for various models (OpenAI GPT-4, OpenAI Whisper-1, Gemini-Pro), including success and failure scenarios.
4.  **`tests/test_prompt_loader.py`**: Unit tests were implemented to confirm the successful loading of prompts from JSON files, and to handle cases of missing files, missing keys, and malformed JSON.
5.  **`tests/test_integration.py`**: Integration tests were implemented to ensure that `transcribe_audio` in `backend/main.py` and `transcribe_audio_chunk` in `worker/worker.py` correctly utilize the `LLMServiceManager` for transcription.

**Errors Fetched and Fixes Applied:**
*   **Dependency Conflicts**: Resolved conflicts with `numpy`, `google-cloud-aiplatform`, and `vertexai` by pinning compatible versions in `backend/requirements.txt`.
*   **GCP Credential Errors (`DefaultCredentialsError`)**: Addressed by implementing lazy initialization for Pub/Sub clients and robust mocking of GCP authentication in `tests/conftest.py`.
*   **`ModuleNotFoundError: No module named 'google.cloud.aiplatform.generative_models'`**: Fixed by ensuring `vertexai` was correctly installed and by adjusting the import path in `llm_service.py` to `from vertexai.preview.generative_models import GenerativeModel, Part`.
*   **`NameError: name 'patch' is not defined`**: Corrected by adding `from unittest.mock import patch` in `tests/test_prompt_loader.py`.
*   **`'bytes' object has no attribute 'name'`**: Fixed in `tests/test_integration.py` by ensuring `io.BytesIO` objects with a `name` attribute were passed to `transcribe_audio_chunk` in tests.
*   **`object AsyncMock can't be used in 'await' expression`**: Resolved in `tests/test_llm_service.py` by explicitly setting `mock_openai_instance.audio.transcriptions.create` to an `AsyncMock` whose `return_value` is the `MagicMock` containing the expected text, ensuring proper awaitable behavior.

All 17 tests passed successfully after applying these fixes.

**Conclusion Statement:**
The implementation and testing of Phase 3.6: Core LLM & Prompt Management Modularity are now complete. This foundational work has established a robust, modular, and testable architecture for integrating various LLMs and managing prompts, paving the way for future AI features.

**Key Considerations for Future Reference:**
*   The `supabase` package deprecation warning should be addressed in a future task by migrating to `supabase_auth`.
*   The `GOOGLE_APPLICATION_CREDENTIALS` environment variable still requires manual user action for local development/testing with actual GCP services.	
	
	

### **Phase 4: Action Item Extraction & Jira Integration**

*   **Objective:** Identify tasks from all sources and connect to the Jira API to create drafts.
*   **Timeline:** 3 Weeks

**4.1 Action Item NLP Model**
[ ] Fine-tune an LLM to identify phrases indicating a task, deadline, or assignment from any text source.
[ ] Extract structured data: {Task, Assignee, Due Date}.

**4.2 Secure Jira Connection**
[ ] Implement OAuth for secure, per-user authentication with the Jira Cloud API.
[ ] Store user tokens securely.

**4.3 Dynamic Field Mapping**
[ ] Develop logic to map extracted info to Jira fields.
[ ] Use channel context (from messages and docs) to intelligently suggest Project, Issue Type, and Priority.

**4.4 Test Draft Generation**
[ ] Build the API call to create a Jira issue but keep it in a "draft" or "to-do" state initially.

### **Phase 5: In-Slack User Approval Workflow**

*   **Objective:** Build the interactive UI for users to approve tickets within Slack DMs.
*   **Timeline:** 2 Weeks

**5.1 Interactive Card Design**
[ ] Use Slack's Block Kit to design a visually clear card for ticket proposals.
[ ] Card must display all key Jira fields.

**5.2 Edit Functionality**
[ ] Implement an "Edit" button on the drafted ticket message.
[ ] Clicking "Edit" will open a Slack Modal pre-filled with the current ticket details (Summary, Description, Assignee, etc.).
[ ] Implement a submission handler for the modal that updates the original message with the edited content.

**5.3 Approval/Discard Logic**
[ ] Handle button clicks for "Create Ticket," "Edit," and "Discard."
[ ] Manage the state of the interactive message (e.g., disable buttons after action).

### **Phase 6: Final Jira Ticket Creation & Deployment**

*   **Objective:** Finalize the workflow by pushing tickets to Jira and preparing for production.
*   **Timeline:** 2 Weeks

**6.1 Push to Jira on Approval**
[ ] Trigger the final Jira API create issue call when a user clicks "Create Ticket."

**6.2 Confirmation & Linkback**
[ ] On successful creation, retrieve the new issue URL.
[ ] Update the original Slack message with a "Ticket Created!" confirmation and a link.

**6.3 Error Handling**
[ ] Implement robust error handling for API failures (Jira or Slack).
[ ] Provide clear, user-friendly error messages.

**6.4 Deployment & Logging**
[ ] Deploy the application to a production-ready cloud environment (e.g., Vercel).
[ ] Configure structured logging and monitoring.

### **Phase 7: Conversational AI & Project Q&A**

*   **Objective:** Transform the bot into an interactive project assistant that can answer questions and provide status updates based on its comprehensive knowledge base.
*   **Timeline:** 4 Weeks

**7.1 Semantic Search & Retrieval**
[ ] Develop a retrieval function that takes a user's question, converts it to an embedding, and performs a similarity search against the vector database to find the most relevant text chunks (the "context").

**7.2 RAG Prompt Engineering**
[ ] Engineer prompts for the Gemini LLM that combine the user's question with the retrieved context (Retrieval-Augmented Generation).
[ ] Instruct the model to answer the question *only* using the provided context to ensure factual accuracy.

**7.3 Jira Status Integration**
[ ] Augment the retrieval logic to also query the Jira API for a ticket's current status if a question is task-related.
[ ] Combine the live Jira status with conversational context from Slack for a complete answer.

**7.4 Conversation Handling**
[ ] Implement an `app_mention` event handler to allow users to trigger conversational Q&A by mentioning the bot in a channel.
[ ] Implement logic to handle multi-turn conversations and follow-up questions in a DM thread.

### **Phase 8: Commercialization & Marketplace Readiness**

*   **Objective:** To transform the bot from a single-instance tool into a secure, multi-tenant SaaS application ready for submission to the Slack App Marketplace.
*   **Timeline:** 4 Weeks

**8.1 Implement Multi-Tenancy**
[x] Update database schema to include a `workspace_id` on all relevant tables.
[x] Implement and enforce strict Row Level Security (RLS) policies in Supabase to ensure data isolation between workspaces.
Implementation Summary: The `workspace_id` is included in `authorized_users`, `slack_messages`, `slack_files`, and `document_embeddings` tables. RLS policies have been implemented for all these tables, enforcing data isolation by `workspace_id` and `channel_id` (for `document_embeddings`). RLS policies were updated to use `x-workspace-id` and `x-channel-id` custom HTTP headers for backend-initiated Supabase calls, removing the need for application-signed JWTs. New `workspace_channels` and `workspace_subscriptions` tables have been created with RLS policies to support channel-level project isolation and subscription tiers. This involved the creation of `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql`.

**8.2 Subscription & Billing Integration**
[x] Create `workspace_subscriptions` table to store subscription details and create RLS policies.
Implementation Summary: The `workspace_subscriptions` table has been created in `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql` to store subscription details, including `plan_id`, `max_channels_allowed`, and `current_channels_count`. RLS policies are in place to secure this table, allowing only the service role to modify it.
[ ] Define subscription tiers (e.g., Free, Pro) with feature limits (e.g., monthly transcription limits).
**Subscription Model Definition:**
- **Tier 1:** 2 projects, 5 users, 5 hours transcription.
  - Monthly Price: $99/mo
  - Yearly Price: $999/y
- **Tier 2:** 5 projects, 25 users, 5 hours transcription.
  - Monthly Price: $399/mo
  - Yearly Price: $3999/y
- **Add-ons (applicable to both tiers):**
  - Additional Project: $49/mo or $499/y
  - Additional User: $5/mo or $50/y
  - Additional Transcription Hour: TBD
[ ] Integrate with Polar.sh for Subscription Management.
**Implementation Plan:**
- **Polar.sh Setup:** Configure the two main subscription tiers and all add-on products (additional projects, users, transcription hours) with their monthly and yearly pricing in the Polar.sh dashboard.
- **Subscription Command:** Create a new slash command (e.g., `/bot-subscribe`) that provides users with a direct link to the Polar.sh subscription page. This link should include the Slack `workspace_id` as metadata to link the payment to the correct workspace.
- **Webhook Handler:** [x] Implement a new Netlify Function (e.g., `polar-webhook.js`) to securely receive and process webhooks from Polar.
- **Database Updates:** [x] The webhook handler will listen for `subscription.created` events. Upon receiving these events, it will update a `founding_members` table in the Supabase database.
**Implementation Summary:**
- Created a `package.json` in `netlify/functions` to manage dependencies (`@polar-sh/sdk`, `@supabase/supabase-js`).
- Created a database migration (`migrations/0006_create_founding_members_table.sql`) for the `founding_members` table.
- Implemented the `netlify/functions/polar-webhook.js` function to handle incoming webhooks, verify them using a secret key, and insert new founding members' data into the Supabase table.
- Removed the now-redundant `/polar/webhooks` endpoint from the main Python application (`main.py`).
[ ] Implement logic to check a workspace's subscription status and enforce feature limits.
[ ] Implement `/bot-enable-channel [channel_name]` slash command to enable a channel for bot interaction and context storage, checking against subscription limits.
[ ] Implement `/bot-disable-channel [channel_name]` slash command to disable a channel, freeing up a subscription slot.

**8.3 Onboarding & User Experience**
[x] Create a friendly welcome message on first install.
[x] Develop a clear onboarding flow that guides admins on how to grant user permissions.
**Implementation Summary:**
- **Intelligent Greeting:** The placeholder response in the `handle_message` function was replaced with an intelligent greeting. The bot now detects simple greetings ("hello", "hi", "hey") from authorized users and responds with a helpful welcome message that introduces its capabilities and prompts the user for a task.


**8.4 Marketplace Submission Prep**
[ ] Create a public-facing landing page with a Privacy Policy, ToS, and support contact.
[ ] Prepare all required materials for Slack's security review process.

### **Phase 9: Continuous Improvement & Advanced Features**

*   **Objective:** Refine the bot's intelligence and add value-add features post-launch.
*   **Timeline:** Ongoing

**9.1 Feedback Mechanism**
[ ] Create slash commands (/bot-feedback) for users to report inaccuracies in Q&A or ticket generation.
[ ] Store feedback to guide model improvements.

**9.2 Global Help Command**
[ ] Implement a global `/help` slash command.
[ ] The command will take a user's question as input.
[ ] It will forward the user's name and question to a designated helpdesk bot or support channel.
[ ] It will send a private confirmation message to the user.
[ ] This command will be available to all users, regardless of authorization status.

**9.3 Self-Learning Context**
[ ] Use approved/edited tickets and feedback on Q&A as training data to improve the accuracy of the context engine and field suggestions.

**9.4 Multi-Channel Support**
[x] Refactor the context engine to maintain separate vector knowledge bases for each private group DM the bot is in.
Implementation Summary: The `document_embeddings` table now includes `channel_id`, and the `workspace_channels` table has been created to explicitly track enabled channels as projects. The `is_authorized` middleware in `main.py` checks if a channel is enabled before processing events, ensuring project-specific knowledge bases and interactions. This also involved the creation of `migrations/0004_add_channel_id_to_embeddings_and_channels_table.sql`.

**9.5 Founders' Circle License System** (PROBLEMATIC)
*   **Objective:** Implement a robust license key generation, distribution, and activation system for Founding Members.
*   **Key Components:**
    *   **Database:** Create `license_keys` table in Supabase for storing unique license keys, customer emails, tier, and redemption status.
    *   **Backend (Pre-Launch):** Develop a FastAPI endpoint to handle Polar.sh payment webhooks, generate UUID-based license keys, store them, and trigger confirmation emails.
    *   **Email:** Integrate an email service (e.g., SendGrid) to send unique license keys, activation instructions, and community links to new Founders.
    *   **UI/UX (Post-Launch):** Design and build in-Slack activation methods (App Home button and `/clarity-activate` slash command) with a modal for key submission.
    *   **Backend (Post-Launch):** Develop a FastAPI endpoint to validate submitted license keys, update `workspace_subscriptions` to the Founder tier, mark keys as redeemed, and send in-Slack confirmations.
    *   **Error Handling:** Implement robust error handling for all stages of the license system.
*   **Additional Actions (Addressing Typo Scenario):**
    *   **Database Schema Update:** Add `client_reference_id` column to the `license_keys` table to link to the original pre-payment user data.
    *   **Netlify Function Update:** Modify `netlify/functions/polar-license-webhook.js` to extract and store the `client_reference_id` from the Polar webhook payload when inserting into the `license_keys` table.



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
