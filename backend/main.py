from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from supabase import create_client, Client, ClientOptions
from slack_bolt.response.response import BoltResponse # Import BoltResponse directly from its submodule
import logging
import io
import httpx
import asyncio # Import asyncio for running async functions in a sync context
import uuid # For generating unique license keys
from datetime import datetime, timezone # Import datetime and timezone
import json # For serializing job payloads

# Google Cloud Pub/Sub for job queuing
from google.cloud import pubsub_v1

# Document parsing libraries

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client with service role key
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_service_role_key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# Use AsyncClient for async operations in middleware and event handlers
supabase: Client = create_client(supabase_url, supabase_service_role_key)

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
PUBSUB_TRANSCRIPTION_TOPIC_NAME = "clarity-transcription-jobs" # Name of the Pub/Sub topic for transcription
PUBSUB_EMBEDDING_TOPIC_NAME = "clarity-embedding-jobs" # Name of the Pub/Sub topic for embeddings
PUBSUB_MESSAGE_TOPIC_NAME = "clarity-message-jobs" # Name of the Pub/Sub topic for general messages

# Lazy initialization for Pub/Sub publisher clients
_pubsub_transcription_publisher = None
_pubsub_embedding_publisher = None
_pubsub_message_publisher = None
_pubsub_transcription_topic_path = None
_pubsub_embedding_topic_path = None
_pubsub_message_topic_path = None

def get_pubsub_transcription_publisher_client():
    global _pubsub_transcription_publisher, _pubsub_transcription_topic_path
    if _pubsub_transcription_publisher is None:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Pub/Sub will not function.")
            return None, None
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub transcription publisher will not function.")
            return None, None
        try:
            _pubsub_transcription_publisher = pubsub_v1.PublisherClient()
            _pubsub_transcription_topic_path = _pubsub_transcription_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TRANSCRIPTION_TOPIC_NAME)
            logger.info("Pub/Sub transcription publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub transcription publisher: {e}")
            return None, None
    return _pubsub_transcription_publisher, _pubsub_transcription_topic_path

def get_pubsub_embedding_publisher_client():
    global _pubsub_embedding_publisher, _pubsub_embedding_topic_path
    if _pubsub_embedding_publisher is None:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Pub/Sub will not function.")
            return None, None
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub embedding publisher will not function.")
            return None, None
        try:
            _pubsub_embedding_publisher = pubsub_v1.PublisherClient()
            _pubsub_embedding_topic_path = _pubsub_embedding_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_EMBEDDING_TOPIC_NAME)
            logger.info("Pub/Sub embedding publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub embedding publisher: {e}")
            return None, None
    return _pubsub_embedding_publisher, _pubsub_embedding_topic_path

def get_pubsub_message_publisher_client():
    global _pubsub_message_publisher, _pubsub_message_topic_path
    if _pubsub_message_publisher is None:
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Pub/Sub will not function.")
            return None, None
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub message publisher will not function.")
            return None, None
        try:
            _pubsub_message_publisher = pubsub_v1.PublisherClient()
            _pubsub_message_topic_path = _pubsub_message_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_MESSAGE_TOPIC_NAME)
            logger.info("Pub/Sub message publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub message publisher: {e}")
            return None, None
    return _pubsub_message_publisher, _pubsub_message_topic_path

# Import LLM Service Manager and Prompt Loader
from services.llm_service import llm_service_manager
from utils.prompt_loader import load_prompt

# Define max file size for direct transcription (25MB for Whisper API)
MAX_DIRECT_TRANSCRIPTION_SIZE_MB = 20
MAX_DIRECT_TRANSCRIPTION_SIZE_BYTES = MAX_DIRECT_TRANSCRIPTION_SIZE_MB * 1024 * 1024

# Initialize Slack Bolt App
slack_app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    process_before_response=True # Acknowledge events immediately
)
slack_handler = AsyncSlackRequestHandler(slack_app)

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/slack/events")
async def endpoint(req: Request):
    return await slack_handler.handle(req)

# --- Slack Event Listeners ---
@slack_app.message()
async def handle_message(message):
    """
    This handler receives all message events.
    It immediately publishes the event to a Pub/Sub topic for background processing.
    """
    publisher, topic_path = get_pubsub_message_publisher_client()
    if publisher and topic_path:
        try:
            # The message object from slack_bolt is not directly JSON serializable
            # We'll convert it to a dict before publishing
            message_payload = {
                "team_id": message.get("team"),
                "channel_id": message.get("channel"),
                "user_id": message.get("user"),
                "message_ts": message.get("ts"),
                "text": message.get("text"),
                "event_ts": message.get("event_ts"),
                "channel_type": message.get("channel_type"),
                "raw_message": message # Include the raw message for the worker
            }
            future = publisher.publish(topic_path, json.dumps(message_payload).encode("utf-8"))
            future.result() # Wait for the publish call to complete
            logger.info(f"Published message event for {message.get('ts')} to Pub/Sub topic: {topic_path}")
        except Exception as e:
            logger.error(f"Error publishing message event to Pub/Sub: {e}")
    else:
        logger.error("Pub/Sub message publisher not configured. Cannot process message event.")

# All processing logic is now handled by the GCP worker.
# This file is only responsible for receiving events and publishing them to Pub/Sub.
