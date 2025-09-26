import os
import json
import asyncio
import io
import httpx
import logging
from dotenv import load_dotenv
import base64

from fastapi import FastAPI, Request, Response, HTTPException

from pydub import AudioSegment
from pydub.silence import split_on_silence

# Document parsing libraries
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
import openpyxl
import pandas as pd

# Vectorization
from sentence_transformers import SentenceTransformer

from supabase import create_client, Client, ClientOptions

# Google Cloud Pub/Sub for job queuing
from google.cloud import pubsub_v1

# Import LLM Service Manager and Prompt Loader
from backend.services.llm_service import llm_service_manager
from backend.utils.prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client with service role key
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_service_role_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_service_role_key)

# Define chunk size for Whisper API (max 25MB)
WHISPER_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024 # 25 MB

app = FastAPI()

@app.post("/")
async def receive_message(request: Request):
    """Receives and processes a push message from a Pub/Sub subscription."""
    envelope = await request.json()
    if not envelope or "message" not in envelope:
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub message format")

    pubsub_message = envelope["message"]
    message_data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
    job_payload = json.loads(message_data)

    # Determine the job type based on the subscription name in the message attributes
    # This is a more robust way to handle multiple job types in one service
    subscription_identifier = pubsub_message.get("attributes", {}).get("googclient_subscription")
    
    if PUBSUB_MESSAGE_SUBSCRIPTION_NAME in subscription_identifier:
        await process_message_job(job_payload)
    elif PUBSUB_TRANSCRIPTION_SUBSCRIPTION_NAME in subscription_identifier:
        await process_transcription_job(job_payload)
    elif PUBSUB_EMBEDDING_SUBSCRIPTION_NAME in subscription_identifier:
        await process_embedding_job(job_payload)
    elif PUBSUB_DOCUMENT_PROCESSING_SUBSCRIPTION_NAME in subscription_identifier:
        await process_document_processing_job(job_payload)
    else:
        logger.warning(f"Unhandled subscription source: {subscription_identifier}")

    return Response(status_code=204)


async def download_file_from_slack(file_url: str, slack_bot_token: str) -> bytes:
    """Downloads a file from Slack."""
    headers = {'Authorization': f'Bearer {slack_bot_token}'}
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, headers=headers)
        response.raise_for_status()
        return response.content

# ... [rest of the helper functions: split_audio_into_chunks, transcribe_audio_chunk, etc.]
# ... [These functions remain unchanged]

# --- Job Processing Functions ---

async def process_message_job(job_payload: dict):
    """Processes a single message job from the queue."""
    logger.info(f"Worker: Processing message job: {job_payload.get('text')}")
    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_bot_token:
        logger.error("SLACK_BOT_TOKEN not set in environment.")
        return

    team_id = job_payload.get("team_id")
    channel_id = job_payload.get("channel_id")
    user_id = job_payload.get("user_id")
    message_ts = job_payload.get("message_ts")
    message_text = job_payload.get("text")
    raw_message = job_payload.get("raw_message")

    if not all([team_id, channel_id, user_id, message_ts, raw_message]):
        logger.error(f"Worker: Invalid message job payload: {job_payload}")
        return

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)
        
        # Re-implementing the authorization logic here
        # A more advanced implementation would share this code with the main app
        is_authorized = await is_user_authorized(rls_supabase_client, team_id, user_id)
        if not is_authorized:
            logger.warning(f"Unauthorized user {user_id} in DM. No reply sent.")
            await send_slack_message(user_id, user_id, "You are not authorized to interact with this bot.", slack_bot_token)
            return

        await asyncio.to_thread(
            rls_supabase_client.from_('slack_messages').insert({
                'slack_message_ts': message_ts,
                'channel_id': channel_id,
                'user_id': user_id,
                'workspace_id': team_id,
                'message_text': message_text,
                'raw_message_data': raw_message
            }).execute
        )
        logger.info(f"Message {message_ts} stored in Supabase.")

        if message_text and message_text.lower().strip() in ["hello", "hi", "hey"]:
            reply_text = ("Hello! I'm the Slack Project Manager bot. I can help you with things like:\n"
                          "*   Answering questions about the project status.\n"
                          "*   Finding technical details from our documents.\n"
                          "*   Creating Jira tickets from conversations.\n\n"
                          "How can I help you today?")
            await send_slack_message(channel_id, user_id, reply_text, slack_bot_token)
        else:
            await send_slack_message(channel_id, user_id, f"Message received and processed: '{message_text}'", slack_bot_token)

    except Exception as e:
        logger.error(f"Worker: Error processing message job for {message_ts}: {e}")

# ... [The rest of the file remains, including all helper and other job processing functions]
# ... [is_user_authorized, process_transcription_job, etc.]
