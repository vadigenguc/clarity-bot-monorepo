import os
import json
import asyncio
import io
import httpx
import logging
from dotenv import load_dotenv
import base64
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response, HTTPException

from supabase import create_client, Client, ClientOptions

# Document parsing libraries (ensure these are in worker/requirements.txt)
import pandas as pd
from docx import Document
import pdfplumber
from pydub import AudioSegment


# Import LLM Service Manager and Prompt Loader
from backend.services.llm_service import llm_service_manager
from backend.utils.prompt_loader import load_prompt

# Google Cloud Pub/Sub for publishing new jobs from the worker
from google.cloud import pubsub_v1
from google.oauth2 import service_account


def get_gcp_credentials():
    """Constructs GCP credentials from environment variables."""
    gcp_credentials_json = os.environ.get("GCP_CREDENTIALS_JSON")
    if gcp_credentials_json:
        try:
            credentials_info = json.loads(gcp_credentials_json)
            return service_account.Credentials.from_service_account_info(credentials_info)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse GCP_CREDENTIALS_JSON: {e}")
            return None
    logger.info("GCP_CREDENTIALS_JSON not found. Falling back to default credentials.")
    return None # Fallback to default credential discovery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client with service role key
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_service_role_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_service_role_key)

# --- Pub/Sub Publisher Setup ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
PUBSUB_TRANSCRIPTION_TOPIC_NAME = "clarity-transcription-jobs"
PUBSUB_EMBEDDING_TOPIC_NAME = "clarity-embedding-jobs"
PUBSUB_MESSAGE_TOPIC_NAME = "clarity-message-jobs"

_pubsub_transcription_publisher = None
_pubsub_embedding_publisher = None
_pubsub_message_publisher = None
_pubsub_transcription_topic_path = None
_pubsub_embedding_topic_path = None
_pubsub_message_topic_path = None

def get_pubsub_transcription_publisher_client():
    global _pubsub_transcription_publisher, _pubsub_transcription_topic_path
    if _pubsub_transcription_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub transcription publisher will not function.")
            return None, None
        try:
            credentials = get_gcp_credentials()
            _pubsub_transcription_publisher = pubsub_v1.PublisherClient(credentials=credentials)
            _pubsub_transcription_topic_path = _pubsub_transcription_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TRANSCRIPTION_TOPIC_NAME)
            logger.info("Pub/Sub transcription publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub transcription publisher: {e}")
            return None, None
    return _pubsub_transcription_publisher, _pubsub_transcription_topic_path

def get_pubsub_embedding_publisher_client():
    global _pubsub_embedding_publisher, _pubsub_embedding_topic_path
    if _pubsub_embedding_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub embedding publisher will not function.")
            return None, None
        try:
            credentials = get_gcp_credentials()
            _pubsub_embedding_publisher = pubsub_v1.PublisherClient(credentials=credentials)
            _pubsub_embedding_topic_path = _pubsub_embedding_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_EMBEDDING_TOPIC_NAME)
            logger.info("Pub/Sub embedding publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub embedding publisher: {e}")
            return None, None
    return _pubsub_embedding_publisher, _pubsub_embedding_topic_path

def get_pubsub_message_publisher_client():
    global _pubsub_message_publisher, _pubsub_message_topic_path
    if _pubsub_message_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub message publisher will not function.")
            return None, None
        try:
            credentials = get_gcp_credentials()
            _pubsub_message_publisher = pubsub_v1.PublisherClient(credentials=credentials)
            _pubsub_message_topic_path = _pubsub_message_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_MESSAGE_TOPIC_NAME)
            logger.info("Pub/Sub message publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub message publisher: {e}")
            return None, None
    return _pubsub_message_publisher, _pubsub_message_topic_path

# Define max file size for direct transcription
MAX_DIRECT_TRANSCRIPTION_SIZE_MB = 20
MAX_DIRECT_TRANSCRIPTION_SIZE_BYTES = MAX_DIRECT_TRANSCRIPTION_SIZE_MB * 1024 * 1024

app = FastAPI()

# --- Main Job Receiver ---
@app.post("/")
async def receive_message(request: Request):
    """Receives and processes a push message from a Pub/Sub subscription."""
    envelope = await request.json()
    if not envelope or "message" not in envelope:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub message format")

    pubsub_message = envelope["message"]
    message_data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
    job_payload = json.loads(message_data)
    job_type = job_payload.get("type")
    
    logger.info(f"Worker: Received job of type '{job_type}'")

    # Route job to the appropriate processor
    job_processors = {
        "message": process_message_job,
        "admin_command": process_admin_command_job,
        "app_home_opened": process_app_home_opened_job,
        "file_shared": process_file_shared_job,
        "file_shared_event": process_file_shared_event_job,
        "activate_license": process_activate_license_job,
        "embedding": process_embedding_job
    }
    
    processor = job_processors.get(job_type)
    if processor:
        await processor(job_payload)
    else:
        logger.warning(f"Worker: Unknown job type received: {job_type}")

    return Response(status_code=204)

# --- Slack Communication Helpers ---
async def send_slack_message(channel_id: str, text: str, slack_bot_token: str, thread_ts: str = None):
    """Sends a message to a Slack channel or user, optionally in a thread."""
    json_payload = {"channel": channel_id, "text": text}
    if thread_ts:
        json_payload["thread_ts"] = thread_ts
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {slack_bot_token}"},
                json=json_payload
            )
            response.raise_for_status()
            if not response.json().get("ok"):
                logger.error(f"Slack API error sending message to {channel_id}: {response.json().get('error')}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error sending Slack message to {channel_id}: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error sending Slack message to {channel_id}: {e}")

async def send_response(response_url: str, text: str, is_ephemeral: bool = True):
    """Sends a response to a Slack command's response_url."""
    payload = {"text": text, "response_type": "ephemeral" if is_ephemeral else "in_channel"}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(response_url, json=payload)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error sending response to {response_url}: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Error sending response to {response_url}: {e}")

# --- Authorization Logic (Ported and adapted for worker) ---
async def is_channel_enabled(rls_supabase_client: Client, team_id: str, channel_id: str) -> bool:
    if not (channel_id.startswith('C') or channel_id.startswith('G')):
        return True
    response = await asyncio.to_thread(
        rls_supabase_client.from_('workspace_channels').select('is_enabled').eq('workspace_id', team_id).eq('channel_id', channel_id).execute
    )
    channel_config = response.data
    if not channel_config or not channel_config[0]['is_enabled']:
        logger.info(f"Ignoring event from disabled channel {channel_id} in workspace {team_id}.")
        return False
    return True

async def is_user_authorized(rls_supabase_client: Client, team_id: str, user_id: str) -> bool:
    response = await asyncio.to_thread(
        rls_supabase_client.from_('authorized_users').select('user_id').eq('workspace_id', team_id).eq('user_id', user_id).execute
    )
    return bool(response.data)

async def are_all_group_members_authorized(rls_supabase_client: Client, team_id: str, channel_id: str, slack_bot_token: str) -> bool:
    if not channel_id.startswith('G'):
        return True
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://slack.com/api/conversations.members",
                headers={"Authorization": f"Bearer {slack_bot_token}"},
                params={"channel": channel_id}
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("ok"): return False
            conversation_members = data.get("members", [])
        
        all_authorized_users_response = await asyncio.to_thread(
            rls_supabase_client.from_('authorized_users').select('user_id').eq('workspace_id', team_id).execute
        )
        all_authorized_user_ids = {u['user_id'] for u in all_authorized_users_response.data}
        
        return all(member_id in all_authorized_user_ids for member_id in conversation_members)
    except Exception as e:
        logger.error(f"Error checking group members' authorization in channel {channel_id}: {e}")
        return False

async def check_authorization(job_payload: dict, slack_bot_token: str) -> tuple[bool, Client | None, str | None]:
    """Main authorization function for the worker."""
    channel_id = job_payload.get("channel_id")
    user_id = job_payload.get("user_id")
    team_id = job_payload.get("team_id")
    message_ts = job_payload.get("message_ts") # For threaded replies

    if not all([channel_id, user_id, team_id]):
        logger.warning(f"Missing context in job payload: {job_payload}")
        # Send generic error to thread if possible
        if channel_id and message_ts:
            await send_slack_message(channel_id, f"Hey <@{user_id}>, your request needs attention, please check your DM.", slack_bot_token, message_ts)
        return False, None, "Error: Missing context information."

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

        if not await is_channel_enabled(rls_supabase_client, team_id, channel_id):
            await send_slack_message(channel_id, f"Hey <@{user_id}>, your request needs attention, please check your DM.", slack_bot_token, message_ts)
            return False, None, "This channel is not enabled for bot interaction."

        user_is_authorized = await is_user_authorized(rls_supabase_client, team_id, user_id)
        
        if channel_id.startswith('D') and not user_is_authorized:
            # No thread for DMs, so no generic message
            return False, None, "You are not authorized to interact with this bot."

        if not await are_all_group_members_authorized(rls_supabase_client, team_id, channel_id, slack_bot_token):
            await send_slack_message(channel_id, f"Hey <@{user_id}>, your request needs attention, please check your DM.", slack_bot_token, message_ts)
            return False, None, "This group chat contains unauthorized members."
        
        # For public/private channels, the user must be authorized
        if (channel_id.startswith('C') or channel_id.startswith('G')) and not user_is_authorized:
            await send_slack_message(channel_id, f"Hey <@{user_id}>, your request needs attention, please check your DM.", slack_bot_token, message_ts)
            return False, None, "You are not authorized to interact with this bot in this channel."

        return True, rls_supabase_client, None
    except Exception as e:
        logger.error(f"An internal error occurred during authorization: {e}")
        await send_slack_message(channel_id, f"Hey <@{user_id}>, your request needs attention, please check your DM.", slack_bot_token, message_ts)
        return False, None, f"An internal error occurred during authorization: {e}"

# --- Document Processing and Vectorization ---
def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_file(file_content: bytes, file_type: str) -> str:
    text = ""
    try:
        if file_type == 'pdf':
            text = extract_text_from_pdf(io.BytesIO(file_content))
        elif file_type == 'docx':
            document = Document(io.BytesIO(file_content))
            text = "\n".join([p.text for p in document.paragraphs])
        elif file_type in ['xlsx', 'csv']:
            df = pd.read_excel(io.BytesIO(file_content)) if file_type == 'xlsx' else pd.read_csv(io.BytesIO(file_content))
            text = df.to_string(index=False)
        elif file_type == 'txt':
            text = file_content.decode('utf-8')
        else:
            logger.warning(f"Unsupported file type for text extraction: {file_type}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_type} file: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    if not text: return []
    words, chunks = text.split(), []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

async def process_and_store_content(workspace_id: str, channel_id: str, source_type: str, source_id: str, content_text: str, rls_supabase_client: Client):
    if not content_text:
        logger.info(f"No content text to process for {source_type} {source_id}.")
        return

    chunks = chunk_text(content_text)
    for i, chunk in enumerate(chunks):
        embedding_job_payload = {
            "type": "embedding",
            "workspace_id": workspace_id, "channel_id": channel_id,
            "source_type": source_type, "source_id": f"{source_id}_chunk_{i}",
            "content": chunk
        }
        publisher, topic_path = get_pubsub_embedding_publisher_client()
        if publisher and topic_path:
            try:
                publisher.publish(topic_path, json.dumps(embedding_job_payload).encode("utf-8"))
            except Exception as e:
                logger.error(f"Error publishing embedding job for chunk {i} of {source_id}: {e}")
        else:
            # If no pub/sub, process directly (for local dev or simplicity)
            await process_embedding_job(embedding_job_payload)

async def transcribe_audio(audio_content: bytes, file_name: str) -> str | None:
    try:
        audio_file = io.BytesIO(audio_content)
        audio_file.name = file_name
        return await llm_service_manager.generate_text(model_name="openai-whisper-1", prompt=audio_file)
    except Exception as e:
        logger.error(f"Error transcribing audio file '{file_name}': {e}")
        return None

async def transcribe_audio_chunk(chunk: io.BytesIO, chunk_index: int) -> str | None:
    """Transcribes a single audio chunk using the LLM service."""
    try:
        chunk.name = f"chunk_{chunk_index}.mp3"
        return await llm_service_manager.generate_text(model_name="openai-whisper-1", prompt=chunk)
    except Exception as e:
        logger.error(f"Error transcribing audio chunk {chunk_index}: {e}")
        return None

async def transcribe_large_audio_file(audio_content: bytes, file_name: str, slack_bot_token: str, channel_id: str, message_ts: str) -> str | None:
    """Splits a large audio file into chunks and transcribes them in parallel."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_content))
        chunk_length_ms = 10 * 60 * 1000  # 10 minutes per chunk
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        
        transcription_tasks = []
        for i, chunk_segment in enumerate(chunks):
            chunk_io = io.BytesIO()
            chunk_segment.export(chunk_io, format="mp3")
            chunk_io.seek(0)
            transcription_tasks.append(transcribe_audio_chunk(chunk_io, i))
            
        await send_slack_message(channel_id, f"Transcribing `{file_name}` in {len(chunks)} chunks. This may take a few minutes...", slack_bot_token, message_ts)
        
        transcriptions = await asyncio.gather(*transcription_tasks)
        full_transcript = " ".join(filter(None, transcriptions))
        return full_transcript
    except Exception as e:
        logger.error(f"Error processing large audio file {file_name}: {e}")
        await send_slack_message(channel_id, f"An error occurred while processing the large audio file `{file_name}`: {e}", slack_bot_token, message_ts)
        return None

# --- Job Processing Functions ---
async def process_message_job(job_payload: dict):
    """Processes a single message job from the queue."""
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    is_authorized, rls_supabase, error_message = await check_authorization(job_payload, slack_bot_token)
    
    if not is_authorized:
        await send_slack_message(job_payload['user_id'], error_message, slack_bot_token)
        return

    team_id, channel_id, user_id = job_payload["team_id"], job_payload["channel_id"], job_payload["user_id"]
    message_ts, message_text, raw_message = job_payload["message_ts"], job_payload.get("text"), job_payload["raw_message"]

    try:
        # If the message is a DM, do not store it. Only provide a conversational response.
        if channel_id.startswith('D'):
            if message_text and message_text.lower().strip() in ["hello", "hi", "hey"]:
                reply_text = ("Hello! I'm the Slack Project Manager bot. How can I help you today?")
                await send_slack_message(channel_id, reply_text, slack_bot_token)
            else:
                # For any other DM, provide a generic response without storing it.
                # This can be enhanced later with conversational AI (Phase 7).
                await send_slack_message(channel_id, "Message received. Note that I only learn from project channels, not DMs.", slack_bot_token)
            return

        # For channel messages, store and process for knowledge base.
        await asyncio.to_thread(
            rls_supabase.from_('slack_messages').insert({
                'slack_message_ts': message_ts, 'channel_id': channel_id, 'user_id': user_id,
                'workspace_id': team_id, 'message_text': message_text, 'raw_message_data': raw_message
            }).execute
        )
        # Correctly pass the rls_supabase client to the processing function
        await process_and_store_content(team_id, channel_id, 'message', message_ts, message_text, rls_supabase)

        # Acknowledge the channel message
        await send_slack_message(channel_id, f"Message received and added to the knowledge base.", slack_bot_token, thread_ts=message_ts)

    except Exception as e:
        logger.error(f"Worker: Error processing message job for {message_ts}: {e}")
        await send_slack_message(channel_id, f"An error occurred while processing your message: {e}", slack_bot_token, thread_ts=message_ts)

async def process_file_shared_event_job(job_payload: dict):
    """
    Handles the raw file_shared event, checks for the 'ingest' keyword,
    and then triggers the actual file processing.
    """
    event = job_payload.get("raw_event", {})
    file_id = event.get('file_id')
    channel_id = event.get('channel_id')
    user_id = event.get('user_id')
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")

    # This is the logic moved from the frontend
    try:
        async with httpx.AsyncClient() as client:
            bot_user_id_res = await client.get("https://slack.com/api/auth.test", headers={"Authorization": f"Bearer {slack_bot_token}"})
            bot_user_id = bot_user_id_res.json().get("user_id")

            history_response = await client.get("https://slack.com/api/conversations.history", headers={"Authorization": f"Bearer {slack_bot_token}"}, params={"channel": channel_id, "limit": 5})
            messages = history_response.json().get('messages', [])
            
            message_with_file = next((msg for msg in messages
                                      if msg.get('user') == user_id and 'files' in msg
                                      and any(f['id'] == file_id for f in msg['files'])), None)

        if message_with_file and f"<@{bot_user_id}>" in message_with_file.get('text', '') and "ingest" in message_with_file.get('text', '').lower():
            # If conditions are met, publish the original 'file_shared' job for processing
            publisher, topic_path = get_pubsub_message_publisher_client()
            if not publisher or not topic_path:
                logger.error("Pub/Sub message publisher not configured for file sharing.")
                return
            
            team_id = event.get("team_id") or (message_with_file.get("files")[0].get("user_team") if message_with_file.get("files") else None)
            processed_payload = {
                "type": "file_shared", # The original job type
                "team_id": team_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "file_id": file_id,
                "raw_event": event,
                "message_ts": message_with_file.get("ts")
            }
            publisher.publish(topic_path, json.dumps(processed_payload).encode("utf-8"))
            
            # Send the initial "I have your file" response
            file_name = message_with_file.get("files")[0].get("name", "your file")
            await send_slack_message(channel_id, f"I have {file_name} and will start processing it now.", slack_bot_token, thread_ts=message_with_file.get("ts"))
        else:
            logger.info(f"Ignoring file {file_id} as bot was not explicitly instructed to ingest.")

    except Exception as e:
        logger.error(f"Error in process_file_shared_event_job for {file_id}: {e}")
        logger.error(traceback.format_exc())

async def process_file_shared_job(job_payload: dict):
    """Processes a file shared event."""
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    is_authorized, rls_supabase, error_message = await check_authorization(job_payload, slack_bot_token)
    
    if not is_authorized:
        await send_slack_message(job_payload['user_id'], error_message, slack_bot_token)
        return

    team_id, channel_id, user_id, file_id = job_payload["team_id"], job_payload["channel_id"], job_payload["user_id"], job_payload["file_id"]
    message_ts = job_payload.get("message_ts") # For threaded replies
    
    try:
        # Idempotency Check: See if this file has already been processed
        existing_file_response = await asyncio.to_thread(
            rls_supabase.from_('slack_files').select('id').eq('slack_file_id', file_id).execute
        )
        if existing_file_response.data:
            logger.warning(f"Duplicate file_shared job received for file {file_id}. Ignoring.")
            return

        # Store initial metadata
        await asyncio.to_thread(
            rls_supabase.from_('slack_files').insert({
                'slack_file_id': file_id, 'channel_id': channel_id, 'user_id': user_id,
                'workspace_id': team_id, 'raw_file_data': job_payload.get("raw_event")
            }).execute
        )

        # Fetch file info and content
        async with httpx.AsyncClient() as client:
            file_info_response = await client.get("https://slack.com/api/files.info", headers={"Authorization": f"Bearer {slack_bot_token}"}, params={"file": file_id})
            file_data = file_info_response.json().get('file')
            if not file_data:
                await send_slack_message(channel_id, f"Could not retrieve info for file `{file_id}`.", slack_bot_token, message_ts)
                return

            file_url, file_type, file_name, file_size = file_data.get('url_private'), file_data.get('filetype'), file_data.get('name'), file_data.get('size', 0)
            
            await asyncio.to_thread(
                rls_supabase.from_('slack_files').update({
                    'file_name': file_name, 'file_type': file_type, 'file_size': file_size, 'file_url': file_url
                }).eq('slack_file_id', file_id).execute
            )

            transcribed_text = None
            if file_type in ['mp3', 'mp4', 'wav', 'm4a', 'mkv', 'webm', 'avi', 'mov']:
                response = await client.get(file_url, headers={"Authorization": f"Bearer {slack_bot_token}"})
                audio_content = await response.aread()
                if file_size > MAX_DIRECT_TRANSCRIPTION_SIZE_BYTES:
                    transcribed_text = await transcribe_large_audio_file(audio_content, file_name, slack_bot_token, channel_id, message_ts)
                else:
                    transcribed_text = await transcribe_audio(audio_content, file_name)
                
                if transcribed_text:
                    await process_and_store_content(team_id, channel_id, 'transcription', file_id, transcribed_text, rls_supabase)
                    await send_slack_message(channel_id, f"I've processed {file_name}. Preparing the summary and analyzing actionable tasks now...", slack_bot_token, message_ts)
                    
                    # Summarization
                    summarization_prompt = load_prompt("summarization_prompt")
                    summary = await llm_service_manager.summarize_text(transcribed_text, summarization_prompt)
                    if summary:
                        await process_and_store_content(team_id, channel_id, 'summary', file_id, summary, rls_supabase)
                        await send_slack_message(channel_id, f"📝 *Summary for `{file_name}`:*\n{summary}", slack_bot_token, message_ts)
                else:
                    await send_slack_message(channel_id, f"❌ Failed to transcribe `{file_name}`.", slack_bot_token, message_ts)
            else:
                response = await client.get(file_url, headers={"Authorization": f"Bearer {slack_bot_token}"})
                file_content = await response.aread()
                # Run the synchronous, potentially long-running text extraction in a separate thread
                extracted_text = await asyncio.to_thread(extract_text_from_file, file_content, file_type)
                if extracted_text:
                    await process_and_store_content(team_id, channel_id, 'file', file_id, extracted_text, rls_supabase)
                    await send_slack_message(channel_id, f"✅ I've processed `{file_name}` and added it to the knowledge base.", slack_bot_token, message_ts)
                else:
                    await send_slack_message(channel_id, f"⚠️ Could not extract text from `{file_name}`.", slack_bot_token, message_ts)
    except Exception as e:
        logger.error(f"Error processing file_shared job for {file_id}: {e}")
        await send_slack_message(channel_id, f"An error occurred while processing your file: {e}", slack_bot_token, message_ts)

async def process_embedding_job(job_payload: dict):
    """Processes an embedding job, generates embedding, and stores it in Supabase."""
    workspace_id = job_payload.get("workspace_id")
    channel_id = job_payload.get("channel_id")
    source_type = job_payload.get("source_type")
    source_id = job_payload.get("source_id")
    content = job_payload.get("content")

    if not all([workspace_id, channel_id, source_type, source_id, content]):
        logger.error(f"Missing data in embedding job payload: {job_payload}")
        return

    try:
        # Generate embedding using the LLM service manager
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('/app/models/all-MiniLM-L6-v2')
        embedding = model.encode(content).tolist()

        # Create an RLS client for the specific workspace and channel
        options = ClientOptions(headers={"x-workspace-id": workspace_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

        # Store the content and embedding in the database
        await asyncio.to_thread(
            rls_supabase_client.from_('document_embeddings').insert({
                'workspace_id': workspace_id,
                'channel_id': channel_id,
                'source_type': source_type,
                'source_id': source_id,
                'content': content,
                'embedding': embedding
            }).execute
        )
        logger.info(f"Successfully processed and stored embedding for {source_type} {source_id}")
    except Exception as e:
        logger.error(f"Error processing embedding job for {source_type} {source_id}: {e}")

async def process_activate_license_job(job_payload: dict):
    """Processes a license activation submission."""
    team_id, user_id = job_payload["team_id"], job_payload["user_id"]
    license_key = job_payload["license_key"]
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")

    try:
        # Use global admin client to check license key table
        license_response = await asyncio.to_thread(
            supabase.from_('license_keys').select('*').eq('license_key', license_key).single().execute
        )
        license_data = license_response.data

        if not license_data:
            await send_slack_message(user_id, "The license key you entered is invalid. Please check the key and try again.", slack_bot_token)
            return
        if license_data['is_redeemed']:
            await send_slack_message(user_id, "This license key has already been redeemed. Please contact support if you believe this is an error.", slack_bot_token)
            return

        # Mark as redeemed and create subscription
        await asyncio.to_thread(
            supabase.from_('license_keys').update({'is_redeemed': True, 'redeemed_by_workspace_id': team_id, 'redeemed_at': datetime.now(timezone.utc).isoformat()}).eq('license_key', license_key).execute
        )
        await asyncio.to_thread(
            supabase.from_('workspace_subscriptions').insert({'workspace_id': team_id, 'plan_id': 'Founder', 'is_active': True}).execute
        )
        
        await send_slack_message(user_id, "🎉 Congratulations! Your Founder membership has been successfully activated. Thank you for your support!", slack_bot_token)
    except Exception as e:
        logger.error(f"Error activating license key for user {user_id}: {e}")
        await send_slack_message(user_id, f"An internal error occurred while activating your license: {e}", slack_bot_token)

# --- Admin Command Helpers (Copied from previous worker version) ---
async def is_user_admin(user_id: str, slack_bot_token: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://slack.com/api/users.info", headers={"Authorization": f"Bearer {slack_bot_token}"}, params={"user": user_id})
            user_info = response.json()
            return user_info.get("ok") and (user_info.get("user", {}).get("is_admin", False) or user_info.get("user", {}).get("is_owner", False))
    except Exception as e:
        logger.error(f"Error checking admin status for user {user_id}: {e}")
        return False

async def find_user_id_by_name(username: str, slack_bot_token: str) -> str | None:
    try:
        async with httpx.AsyncClient() as client:
            cursor = None
            while True:
                response = await client.get("https://slack.com/api/users.list", headers={"Authorization": f"Bearer {slack_bot_token}"}, params={"limit": 200, "cursor": cursor})
                data = response.json()
                if not data.get("ok"): return None
                for user in data.get("members", []):
                    if user.get("name") == username or user.get("profile", {}).get("display_name") == username:
                        return user.get("id")
                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor: break
    except Exception as e:
        logger.error(f"Error looking up user by name '{username}': {e}")
    return None

async def process_admin_command_job(job_payload: dict):
    command_name, requesting_user_id, team_id, response_url, text = job_payload.get("command_name"), job_payload.get("user_id"), job_payload.get("team_id"), job_payload.get("response_url"), job_payload.get("text", "").strip()
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")

    if not await is_user_admin(requesting_user_id, slack_bot_token):
        await send_response(response_url, "Sorry, only workspace admins can run this command.")
        return

    supabase_admin = supabase # Use global client for admin actions

    if command_name == "/bot-list-authorized":
        response = await asyncio.to_thread(supabase_admin.from_('authorized_users').select('user_id').eq('workspace_id', team_id).execute)
        user_mentions = [f"<@{user['user_id']}>" for user in response.data]
        message = f"Authorized users: {', '.join(user_mentions)}" if user_mentions else "No authorized users found."
        await send_response(response_url, message)

    elif command_name in ["/bot-grant-access", "/bot-revoke-access"]:
        if not text:
            await send_response(response_url, f"Usage: `{command_name} @user`")
            return
        parsed_user_id = text.split('|')[0].strip('<@>') if text.startswith('<@U') else await find_user_id_by_name(text.strip('@'), slack_bot_token)
        if not parsed_user_id:
            await send_response(response_url, f"Could not find user `{text}`.")
            return

        if command_name == "/bot-grant-access":
            try:
                await asyncio.to_thread(supabase_admin.from_('authorized_users').insert({'workspace_id': team_id, 'user_id': parsed_user_id}).execute)
                await send_response(response_url, f"Access granted to <@{parsed_user_id}>.")
            except Exception as e:
                await send_response(response_url, f"<@{parsed_user_id}> is already authorized." if "violates unique constraint" in str(e) else f"Error: {e}")
        else: # /bot-revoke-access
            response = await asyncio.to_thread(supabase_admin.from_('authorized_users').delete().eq('workspace_id', team_id).eq('user_id', parsed_user_id).execute)
            await send_response(response_url, f"Access revoked for <@{parsed_user_id}>." if response.data else f"<@{parsed_user_id}> was not authorized.")

async def process_app_home_opened_job(job_payload: dict):
    user_id, team_id, slack_bot_token = job_payload.get("user_id"), job_payload.get("team_id"), os.environ.get("SLACK_BOT_TOKEN")
    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": "none"})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)
        subscription_response = await asyncio.to_thread(rls_supabase_client.from_('workspace_subscriptions').select('plan_id').eq('workspace_id', team_id).single().execute)
        is_founder = subscription_response.data and subscription_response.data['plan_id'] == 'Founder'

        blocks = ([{"type": "section", "text": {"type": "mrkdwn", "text": f"Welcome back, Founder <@{user_id}>! How can I assist?"}}]
                  if is_founder else
                  [{"type": "section", "text": {"type": "mrkdwn", "text": f"Welcome, <@{user_id}>! Unlock your Founder benefits."}},
                   {"type": "actions", "elements": [{"type": "button", "text": {"type": "plain_text", "text": "Activate Membership"}, "style": "primary", "action_id": "open_license_activation_modal"}]},
                   {"type": "section", "text": {"type": "mrkdwn", "text": "Already a Founder? Use `/clarity-activate`."}}])
        
        async with httpx.AsyncClient() as client:
            await client.post("https://slack.com/api/views.publish", headers={"Authorization": f"Bearer {slack_bot_token}"}, json={"user_id": user_id, "view": {"type": "home", "blocks": blocks}})
    except Exception as e:
        logger.error(f"Error processing app_home_opened job for user {user_id}: {e}")
