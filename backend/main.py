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
from google.oauth2 import service_account

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
PUBSUB_ADMIN_TOPIC_NAME = "clarity-admin-jobs" # Name of the Pub/Sub topic for admin commands

# Lazy initialization for Pub/Sub publisher clients
_pubsub_transcription_publisher = None
_pubsub_embedding_publisher = None
_pubsub_message_publisher = None
_pubsub_admin_publisher = None
_pubsub_transcription_topic_path = None
_pubsub_embedding_topic_path = None
_pubsub_message_topic_path = None
_pubsub_admin_topic_path = None

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

def get_pubsub_admin_publisher_client():
    global _pubsub_admin_publisher, _pubsub_admin_topic_path
    if _pubsub_admin_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub admin publisher will not function.")
            return None, None
        try:
            credentials = get_gcp_credentials()
            _pubsub_admin_publisher = pubsub_v1.PublisherClient(credentials=credentials)
            _pubsub_admin_topic_path = _pubsub_admin_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_ADMIN_TOPIC_NAME)
            logger.info("Pub/Sub admin publisher client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub admin publisher: {e}")
            return None, None
    return _pubsub_admin_publisher, _pubsub_admin_topic_path


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
    """Publishes all message events to Pub/Sub for background processing."""
    if message.get("subtype") == "file_share":
        return # Ignore file_share subtypes to avoid duplicate processing
    publisher, topic_path = get_pubsub_message_publisher_client()
    if not publisher or not topic_path:
        logger.error("Pub/Sub message publisher not configured.")
        return

    try:
        payload = {
            "type": "message", "team_id": message.get("team"), "channel_id": message.get("channel"),
            "user_id": message.get("user"), "message_ts": message.get("ts"), "text": message.get("text"),
            "raw_message": message
        }
        future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
        future.result()
    except Exception as e:
        logger.error(f"Error publishing message event to Pub/Sub: {e}")

@slack_app.event("file_shared")
async def handle_file_shared(event, say):
    """
    Handles file_shared events, publishing them for background processing if the bot is mentioned.
    """
    # Acknowledge the event immediately to prevent timeouts, but only if explicitly asked to ingest.
    bot_user_id_res = await slack_app.client.auth_test()
    bot_user_id = bot_user_id_res.get("user_id")
    
    history_response = await slack_app.client.conversations_history(channel=event.get('channel_id'), limit=5)
    message_with_file = next((msg for msg in history_response.get('messages', []) 
                              if msg.get('user') == event.get('user_id') and 'files' in msg 
                              and any(f['id'] == event.get('file_id') for f in msg['files'])), None)

    if message_with_file and f"<@{bot_user_id}>" in message_with_file.get('text', '') and "ingest" in message_with_file.get('text', '').lower():
        await say("Thanks! I've received your file and will start processing it now.")
        
        publisher, topic_path = get_pubsub_message_publisher_client()
        if not publisher or not topic_path:
            logger.error("Pub/Sub message publisher not configured for file sharing.")
            return

        try:
            team_id = event.get("team_id") or (message_with_file.get("files")[0].get("user_team") if message_with_file.get("files") else None)
            payload = {
                "type": "file_shared", "team_id": team_id, "channel_id": event.get("channel_id"),
                "user_id": event.get("user_id"), "file_id": event.get("file_id"), "raw_event": event,
                "message_ts": message_with_file.get("ts")
            }
            future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
            future.result()
        except Exception as e:
            logger.error(f"Error publishing file_shared event to Pub/Sub: {e}")
    else:
        logger.info(f"Ignoring file {event.get('file_id')} as bot was not explicitly instructed to ingest.")

# --- Slack Command Listeners ---
async def publish_admin_command(command):
    """Helper function to publish any admin command to Pub/Sub."""
    publisher, topic_path = get_pubsub_admin_publisher_client()
    if not publisher or not topic_path:
        logger.error("Pub/Sub admin publisher not configured. Cannot process admin command.")
        return

    try:
        command_payload = {
            "type": "admin_command",
            "command_name": command["command"],
            "team_id": command["team_id"],
            "user_id": command["user_id"],
            "channel_id": command["channel_id"],
            "response_url": command["response_url"],
            "trigger_id": command["trigger_id"],
            "text": command.get("text", "")
        }
        future = publisher.publish(topic_path, json.dumps(command_payload).encode("utf-8"))
        future.result()
        logger.info(f"Published admin command {command['command']} to Pub/Sub topic: {topic_path}")
    except Exception as e:
        logger.error(f"Error publishing admin command to Pub/Sub: {e}")

@slack_app.command("/bot-list-authorized")
async def handle_list_authorized_command(ack, command):
    await ack()
    await publish_admin_command(command)

@slack_app.command("/bot-grant-access")
async def handle_grant_access_command(ack, command):
    await ack()
    await publish_admin_command(command)

@slack_app.command("/bot-revoke-access")
async def handle_revoke_access_command(ack, command):
    await ack()
    await publish_admin_command(command)

@slack_app.command("/clarity-activate")
async def handle_clarity_activate_command(ack, body, client):
    await ack()
    await client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal", "callback_id": "activate_license_modal",
            "title": {"type": "plain_text", "text": "Activate Founder Membership"},
            "submit": {"type": "plain_text", "text": "Activate"},
            "blocks": [{
                "type": "input", "block_id": "license_key_input_block",
                "label": {"type": "plain_text", "text": "Enter your Founder License Key"},
                "element": {"type": "plain_text_input", "action_id": "license_key_input"}
            }]
        }
    )

@slack_app.action("open_license_activation_modal")
async def open_license_modal(ack, body, client):
    await ack()
    await client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal", "callback_id": "activate_license_modal",
            "title": {"type": "plain_text", "text": "Activate Founder Membership"},
            "submit": {"type": "plain_text", "text": "Activate"},
            "blocks": [{
                "type": "input", "block_id": "license_key_input_block",
                "label": {"type": "plain_text", "text": "Enter your Founder License Key"},
                "element": {"type": "plain_text_input", "action_id": "license_key_input"}
            }]
        }
    )

@slack_app.view("activate_license_modal")
async def handle_license_activation_submission(ack, body, logger):
    await ack()
    publisher, topic_path = get_pubsub_message_publisher_client()
    if not publisher or not topic_path:
        logger.error("Pub/Sub message publisher not configured for license activation.")
        return

    try:
        license_key = body["view"]["state"]["values"]["license_key_input_block"]["license_key_input"]["value"]
        payload = {
            "type": "activate_license", "user_id": body["user"]["id"], "team_id": body["user"]["team_id"],
            "license_key": license_key
        }
        future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
        future.result()
    except Exception as e:
        logger.error(f"Error publishing license activation to Pub/Sub: {e}")

@slack_app.event("app_home_opened")
async def handle_app_home_opened(event, logger):
    """Publishes app_home_opened events to Pub/Sub for background processing."""
    publisher, topic_path = get_pubsub_message_publisher_client()
    if not publisher or not topic_path:
        logger.error("Pub/Sub message publisher not configured.")
        return

    try:
        payload = {
            "type": "app_home_opened", "user_id": event.get("user"), "team_id": event.get("team"),
            "raw_event": event
        }
        future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
        future.result()
    except Exception as e:
        logger.error(f"Error publishing app_home_opened event to Pub/Sub: {e}")
