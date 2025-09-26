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

# Lazy initialization for Pub/Sub publisher clients
_pubsub_transcription_publisher = None
_pubsub_embedding_publisher = None
_pubsub_transcription_topic_path = None
_pubsub_embedding_topic_path = None

def get_pubsub_transcription_publisher_client():
    global _pubsub_transcription_publisher, _pubsub_transcription_topic_path
    if _pubsub_transcription_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub transcription publisher will not function.")
            return None, None
        _pubsub_transcription_publisher = pubsub_v1.PublisherClient()
        _pubsub_transcription_topic_path = _pubsub_transcription_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TRANSCRIPTION_TOPIC_NAME)
        logger.info("Pub/Sub transcription publisher client initialized.")
    return _pubsub_transcription_publisher, _pubsub_transcription_topic_path

def get_pubsub_embedding_publisher_client():
    global _pubsub_embedding_publisher, _pubsub_embedding_topic_path
    if _pubsub_embedding_publisher is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub embedding publisher will not function.")
            return None, None
        _pubsub_embedding_publisher = pubsub_v1.PublisherClient()
        _pubsub_embedding_topic_path = _pubsub_embedding_publisher.topic_path(GCP_PROJECT_ID, PUBSUB_EMBEDDING_TOPIC_NAME)
        logger.info("Pub/Sub embedding publisher client initialized.")
    return _pubsub_embedding_publisher, _pubsub_embedding_topic_path

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


# @app.post("/webhooks/polar-license")
# async def polar_license_webhook(request: Request):
#     """
#     Handles incoming Polar.sh webhooks for license key generation and distribution.
#     """
#     logger.info("Received Polar license webhook.")
#     payload = await request.body()
#     signature = request.headers.get("polar-signature")
    
#     if not signature:
#         logger.error("Polar-Signature header missing.")
#         raise HTTPException(status_code=400, detail="Polar-Signature header missing.")

#     polar_webhook_secret = os.environ.get("POLAR_WEBHOOK_SECRET")
#     if not polar_webhook_secret:
#         logger.error("POLAR_WEBHOOK_SECRET environment variable not set.")
#         raise HTTPException(status_code=500, detail="Server configuration error.")

#     try:
#         # Verify webhook signature
#         verified_payload = Webhook.verify(payload, signature, polar_webhook_secret)
#         webhook_data = PolarWebhook.parse_raw(verified_payload.json())
#         logger.info(f"Verified Polar webhook of type: {webhook_data.type}")

#         if webhook_data.type == WebhookType.ORDER_SUCCEEDED:
#             customer_email = webhook_data.data.customer_email
#             tier = webhook_data.data.product_name # Assuming product_name maps to tier
            
#             license_key = f"CLARITY-FOUNDER-{uuid.uuid4()}"
#             logger.info(f"Generated license key {license_key} for {customer_email} ({tier}).")

#             # Store license key in Supabase
#             options = ClientOptions(headers={"x-workspace-id": "none", "x-channel-id": "none"}) # No workspace/channel yet
#             rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

#             await asyncio.to_thread(
#                 rls_supabase_client.from_('license_keys').insert({
#                     'license_key': license_key,
#                     'customer_email': customer_email,
#                     'tier': tier,
#                     'is_redeemed': False
#                 }).execute
#             )
#             logger.info(f"License key {license_key} stored in Supabase for {customer_email}.")

#             # TODO: Integrate email service to send the license key to the customer_email
#             # For now, we'll just log it.
#             logger.info(f"Email to {customer_email} with license key {license_key} would be sent here.")
#             # Example: await send_license_key_email(customer_email, license_key, os.environ.get("SLACK_FOUNDER_COMMUNITY_LINK"))

#             return Response(status_code=200)
#         else:
#             logger.info(f"Unhandled Polar webhook type: {webhook_data.type}")
#             return Response(status_code=200) # Acknowledge other webhook types
            
#     except Exception as e:
#         logger.error(f"Error processing Polar webhook: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing webhook: {e}")

# --- Vectorization and Document Processing Functions ---

# --- Vectorization and Document Processing Functions ---

# Placeholder for embedding generation (now handled by worker)
def get_embedding(text: str):
    logger.warning("Embedding generation is offloaded to worker. This function should not be called directly.")
    return [] # Return empty list as placeholder

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Splits text into chunks with optional overlap."""
    if not text:
        return []
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i < 0: # Ensure i doesn't go negative if overlap > chunk_size
            i = 0
    return chunks

def extract_text_from_file(file_content: bytes, file_type: str) -> str:
    """Extracts text from various file types."""
    text = ""
    try:
        if file_type == 'pdf':
            text = extract_text_from_pdf(io.BytesIO(file_content))
        elif file_type == 'docx':
            document = Document(io.BytesIO(file_content))
            text = "\n".join([p.text for p in document.paragraphs])
        elif file_type in ['xlsx', 'csv']:
            if file_type == 'xlsx':
                xls = pd.ExcelFile(io.BytesIO(file_content))
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    text += df.to_string(index=False) + "\n"
            elif file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
                text = df.to_string(index=False)
        elif file_type == 'txt':
            text = file_content.decode('utf-8')
        elif file_type in ['mp3', 'mp4', 'wav', 'm4a']: # Audio/Video files handled by transcription
            logger.info(f"Audio/Video file type '{file_type}' detected. Text extraction skipped, will be transcribed.")
            return "" # Return empty string, content will be transcribed
        else:
            logger.warning(f"Unsupported file type for text extraction: {file_type}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_type} file: {e}")
    return text

async def process_and_store_content(
    workspace_id: str, 
    channel_id: str, # Added channel_id
    source_type: str, 
    source_id: str, 
    content_text: str, 
    rls_supabase_client: Client
):
    if not content_text:
        logger.info(f"No content text to process for {source_type} {source_id} in channel {channel_id}.")
        return

    chunks = chunk_text(content_text)
    for i, chunk in enumerate(chunks):
        # Instead of generating embedding here, publish an embedding job to Pub/Sub
        embedding_job_payload = {
            "workspace_id": workspace_id,
            "channel_id": channel_id,
            "source_type": source_type,
            "source_id": f"{source_id}_chunk_{i}",
            "content": chunk
        }
        
        publisher, topic_path = get_pubsub_embedding_publisher_client()
        if publisher and topic_path:
            try:
                future = publisher.publish(topic_path, json.dumps(embedding_job_payload).encode("utf-8"))
                future.result() # Wait for the publish call to complete
                logger.info(f"Published embedding job for chunk {i} of {source_type} {source_id} to Pub/Sub topic: {topic_path}")
            except Exception as e:
                logger.error(f"Error publishing embedding job for chunk {i} of {source_type} {source_id}: {e}")
        else:
            logger.error(f"Pub/Sub embedding topic path not configured. Cannot offload embedding for chunk {i} of {source_type} {source_id}.")

# --- Authorization Logic ---

async def is_channel_enabled(rls_supabase_client: Client, team_id: str, channel_id: str, logger) -> bool:
    """Check if a public or private channel is enabled for the bot."""
    if not (channel_id.startswith('C') or channel_id.startswith('G')):
        return True # DMs and Group DMs don't need to be "enabled"
    
    response = await asyncio.to_thread(
        rls_supabase_client.from_('workspace_channels').select('is_enabled').eq('workspace_id', team_id).eq('channel_id', channel_id).execute
    )
    channel_config = response.data
    
    if not channel_config or not channel_config[0]['is_enabled']:
        logger.info(f"Ignoring event from disabled channel {channel_id} in workspace {team_id}.")
        return False
    return True

async def is_user_authorized(rls_supabase_client: Client, team_id: str, user_id: str) -> bool:
    """Check if a single user is in the authorized_users table."""
    response = await asyncio.to_thread(
        rls_supabase_client.from_('authorized_users').select('user_id').eq('workspace_id', team_id).eq('user_id', user_id).execute
    )
    return bool(response.data)

async def is_user_admin(user_id: str, client) -> bool:
    """Check if a user is an admin or owner of the workspace."""
    try:
        user_info = await client.users_info(user=user_id)
        return user_info.get("user", {}).get("is_admin", False) or user_info.get("user", {}).get("is_owner", False)
    except Exception as e:
        logger.error(f"Error checking admin status for user {user_id}: {e}")
        return False

async def find_user_id_by_name(username: str, client) -> str | None:
    """Find a user ID by their username or display name by paginating through all users."""
    try:
        cursor = None
        while True:
            response = await client.users_list(cursor=cursor, limit=200)
            for user in response.get("members", []):
                if user.get("name") == username or user.get("profile", {}).get("display_name") == username:
                    return user.get("id")
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except Exception as e:
        logger.error(f"Error looking up user by name '{username}': {e}")
    return None

async def are_all_group_members_authorized(rls_supabase_client: Client, team_id: str, channel_id: str, logger) -> bool:
    """Check if all members of a group DM are authorized."""
    if not channel_id.startswith('G'):
        return True # Not a group DM, so this check doesn't apply

    try:
        conversation_members_response = await slack_app.client.conversations_members(channel=channel_id)
        conversation_members = conversation_members_response.get("members", [])
        
        all_authorized_users_response = await asyncio.to_thread(
            rls_supabase_client.from_('authorized_users').select('user_id').eq('workspace_id', team_id).execute
        )
        all_authorized_user_ids = {u['user_id'] for u in all_authorized_users_response.data}

        all_members_authorized = all(member_id in all_authorized_user_ids for member_id in conversation_members)

        if not all_members_authorized:
            logger.info(f"Ignoring event from group chat with unauthorized members in {channel_id}.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking group members' authorization in channel {channel_id}: {e}")
        return False

async def check_authorization(context, body, logger, say):
    """Main authorization function to orchestrate all checks."""
    channel_id = context.get("channel_id")
    user_id = context.get("user_id")
    team_id = context.get("team_id")
    
    # For slash commands, the IDs are in the body
    if command := body.get("command"):
        channel_id = body.get("channel_id")
        user_id = body.get("user_id")
        team_id = body.get("team_id")

    event_type = body.get("event", {}).get("type")

    if not all([channel_id, user_id, team_id]):
        logger.warning("Missing channel_id, user_id, or team_id in event context.")
        await say("Error: Missing context information. Please try again or contact support.")
        return False

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(
            supabase_url, 
            supabase_service_role_key,
            options=options
        )
        context["rls_supabase"] = rls_supabase_client

        # 1. Check if the channel is enabled
        if not await is_channel_enabled(rls_supabase_client, team_id, channel_id, logger):
            await say("This channel is not enabled for bot interaction. Please contact your workspace admin.")
            return False

        # 2. Check if the initiating user is authorized
        user_is_authorized = await is_user_authorized(rls_supabase_client, team_id, user_id)
        
        # 3. Determine if the event requires authorization
        requires_user_auth = (event_type in ["message", "file_shared", "app_mention"] and not channel_id.startswith('D')) or command
        
        if requires_user_auth and not user_is_authorized:
            logger.info(f"Ignoring interactive event from unauthorized user {user_id} in channel {channel_id}.")
            await say("You are not authorized to interact with this bot. Please contact your workspace admin.")
            return False
        
        # 4. For DMs, the single user must be authorized
        if channel_id.startswith('D') and not user_is_authorized:
            logger.info(f"Ignoring DM event from unauthorized user {user_id}.")
            await say("You are not authorized to interact with this bot.")
            return False

        # 5. For Group DMs, all members must be authorized
        if not await are_all_group_members_authorized(rls_supabase_client, team_id, channel_id, logger):
            await say("This group chat contains unauthorized members. The bot can only interact in group DMs where all members are authorized.")
            return False
        
        return True # All checks passed

    except Exception as e:
        logger.error(f"An internal error occurred during authorization: {e}")
        await say(f"An internal error occurred during authorization: {e}")
        return False

# --- Slack Event Listeners ---
@slack_app.message()
async def handle_message(message, say, logger, context):
    logger.info(f"Message event received: {message}")
    # Call authorization check
    is_authorized = await check_authorization(context, message, logger, say)
    if not is_authorized:
        return

    rls_supabase = context["rls_supabase"]
    team_id = message['team']
    channel_id = message['channel'] # Get channel_id
    message_ts = message['ts']
    message_text = message.get('text')

    try:
        # Store raw message data
        await asyncio.to_thread(
            rls_supabase.from_('slack_messages').insert({
                'slack_message_ts': message_ts,
                'channel_id': channel_id,
                'user_id': message['user'],
                'workspace_id': team_id,
                'message_text': message_text,
                'raw_message_data': message
            }).execute
        )
        logger.info(f"Message {message_ts} stored in Supabase.")

        # Process and store embeddings
        await process_and_store_content(team_id, channel_id, 'message', message_ts, message_text, rls_supabase)

    except Exception as e:
        logger.error(f"Error handling message event or storing data: {e}")

    # Check for a greeting and respond with a helpful message
    if message_text and message_text.lower().strip() in ["hello", "hi", "hey"]:
        await say(
            "Hello! I'm the Slack Project Manager bot. I can help you with things like:\n"
            "*   Answering questions about the project status.\n"
            "*   Finding technical details from our documents.\n"
            "*   Creating Jira tickets from conversations.\n\n"
            "How can I help you today?"
        )
    else:
        # Placeholder for future conversational AI
        await say(f"Message received and processed: '{message_text}'")


@slack_app.event("file_shared")
async def handle_file_shared(event, say, logger, context):
    logger.info(f"File shared event received: {event}")
    # Call authorization check
    is_authorized = await check_authorization(context, event, logger, say)
    if not is_authorized:
        return BoltResponse(status=200) # Return a BoltResponse even for missing context

    rls_supabase = context["rls_supabase"]
    team_id = event.get('team_id')
    file_id = event.get('file_id')
    channel_id = event.get('channel_id')
    user_id = event.get('user_id') # Get user_id from event

    if not file_id or not channel_id or not team_id or not user_id:
        logger.error("Missing file_id, channel_id, team_id, or user_id in file_shared event.")
        return BoltResponse(status=200)

    # Check if the file was shared with a message containing bot mention and "ingest"
    # This requires fetching the message associated with the file_shared event.
    # Slack's file_shared event doesn't directly include the message text.
    # We need to check the latest messages in the channel for a message that mentions the bot and the file.
    try:
        # Fetch recent messages in the channel to find the one associated with the file share
        # We look for messages from the user who shared the file, containing a bot mention and "ingest"
        history_response = await slack_app.client.conversations_history(
            channel=channel_id,
            limit=5 # Look at the last 5 messages
        )
        
        message_text_with_file = None
        for msg in history_response.get('messages', []):
            if msg.get('user') == user_id and 'files' in msg and any(f['id'] == file_id for f in msg['files']):
                message_text_with_file = msg.get('text', '').lower()
                break

        bot_user_id = slack_app.client.auth_test().get("user_id")
        if not bot_user_id:
            logger.error("Could not retrieve bot user ID.")
            await say(f"An internal error occurred. Could not retrieve bot ID.")
            return BoltResponse(status=200)

        bot_mention_pattern = f"<@{bot_user_id}>"
        
        if not message_text_with_file or bot_mention_pattern not in message_text_with_file or "ingest" not in message_text_with_file:
            logger.info(f"Ignoring file {file_id} as bot was not explicitly instructed to ingest.")
            return BoltResponse(status=200) # Not explicitly instructed, so ignore

        # Proceed with ingestion if explicitly instructed
        # Store raw file metadata
        await asyncio.to_thread(
            rls_supabase.from_('slack_files').insert({
                'slack_file_id': file_id,
                'channel_id': channel_id,
                'user_id': user_id,
                'workspace_id': team_id,
                'file_name': "Fetching...",
                'file_type': "Fetching...",
                'file_size': 0,
                'file_url': "Fetching...",
                'raw_file_data': event
            }).execute
        )
        logger.info(f"File {file_id} metadata placeholder stored in Supabase.")

        # Fetch file info using Slack API
        file_info_response = await slack_app.client.files_info(file=file_id)
        file_data = file_info_response.get('file')

        if file_data:
            # Download file content
            file_url = file_data.get('url_private')
            file_type = file_data.get('filetype')
            file_name = file_data.get('name')
            file_size = file_data.get('size', 0) # Get file size

            if file_url and file_type:
                # Update stored file metadata with actual details (before downloading large files)
                await asyncio.to_thread(
                    rls_supabase.from_('slack_files').update({
                        'file_name': file_name,
                        'file_type': file_type,
                        'file_size': file_size,
                        'file_url': file_url,
                        'raw_file_data': file_data
                    }).eq('slack_file_id', file_id).execute
                )
                logger.info(f"File {file_id} metadata updated in Supabase.")

                # Determine if the file is an audio/video file
                if file_type in ['mp3', 'mp4', 'wav', 'm4a']:
                    if file_size > MAX_DIRECT_TRANSCRIPTION_SIZE_BYTES:
                        # For large files, push to a job queue
                        job_payload = {
                            "file_url": file_url,
                            "workspace_id": team_id,
                            "channel_id": channel_id,
                            "file_id": file_id,
                            "file_name": file_name,
                            "user_id": user_id,
                            "file_type": file_type
                        }
                        
                        publisher, topic_path = get_pubsub_transcription_publisher_client()
                        if publisher and topic_path:
                            # Publish the job payload to Pub/Sub
                            future = publisher.publish(topic_path, json.dumps(job_payload).encode("utf-8"))
                            future.result() # Wait for the publish call to complete
                            logger.info(f"Large file ({file_size / (1024*1024):.2f} MB) detected. Pushed transcription job to Pub/Sub topic: {topic_path}")
                            await say(f"⏳ Your large audio/video file `{file_name}` is being processed in the background. I'll notify you when the transcription is complete!")
                        else:
                            logger.error("Pub/Sub transcription topic path not configured. Cannot queue large file for transcription.")
                            await say(f"❌ Failed to process large file `{file_name}`: Server configuration error (Pub/Sub not set up).")
                    else:
                        # For small files, download and transcribe directly
                        headers = {'Authorization': f'Bearer {slack_app.token}'}
                        async with httpx.AsyncClient() as client:
                            response = await client.get(file_url, headers=headers)
                            response.raise_for_status()
                            file_content_bytes = await response.aread()

                        transcribed_text = await transcribe_audio(file_content_bytes, file_name)
                        if transcribed_text:
                            await process_and_store_content(team_id, channel_id, 'transcription', file_id, transcribed_text, rls_supabase)
                            await say(f"✅ I've transcribed `{file_name}` and added it to the project knowledge base.")
                        else:
                            await say(f"❌ Failed to transcribe `{file_name}`.")
                else:
                    # Process as a regular document
                    headers = {'Authorization': f'Bearer {slack_app.token}'}
                    async with httpx.AsyncClient() as client:
                        response = await client.get(file_url, headers=headers)
                        response.raise_for_status()
                        file_content_bytes = await response.aread()

                    extracted_text = extract_text_from_file(file_content_bytes, file_type)
                    if extracted_text:
                        await process_and_store_content(team_id, channel_id, 'file', file_id, extracted_text, rls_supabase)
                        await say(f"✅ I've processed `{file_name}` and added it to the project knowledge base.")
                    else:
                        await say(f"⚠️ Could not extract text from `{file_name}`. It might be an unsupported format or empty.")
            else:
                logger.warning(f"File URL or type missing for file_id: {file_id}")
                await say(f"Could not process file `{file_data.get('name', 'unknown')}`: Missing URL or type.")
        else:
            logger.warning(f"Could not retrieve file info for file_id: {file_id}")
            await say(f"Could not retrieve information for file `{file_id}`.")

    except Exception as e:
        logger.error(f"Error handling file_shared event or storing file metadata: {e}")
        await say(f"An error occurred while processing your file: {e}")

async def transcribe_audio(audio_content: bytes, file_name: str) -> str | None:
    """Transcribes audio content using OpenAI Whisper via LLMServiceManager."""
    try:
        # OpenAI's API expects a file-like object
        audio_file = io.BytesIO(audio_content)
        audio_file.name = file_name # Set a name for the file-like object

        # Use the LLMServiceManager for transcription
        # Assuming 'whisper-1' is an OpenAI model, prefix with 'openai-'
        transcript_text = await llm_service_manager.generate_text(
            model_name="openai-whisper-1", 
            prompt=audio_file # Pass the file-like object directly
        )
        return transcript_text
    except Exception as e:
        logger.error(f"Error transcribing audio file '{file_name}': {e}")
        return None


# --- Slack App Home and Slash Command Listeners ---

@slack_app.event("app_home_opened")
async def handle_app_home_opened(client, event, logger, context):
    user_id = event["user"]
    team_id = event["team"]

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": "none"})
        rls_supabase_client = create_client(
            supabase_url, 
            supabase_service_role_key,
            options=options
        )
        
        # Check if the workspace is a Founder
        subscription_response = await asyncio.to_thread(
            rls_supabase_client.from_('workspace_subscriptions').select('plan_id').eq('workspace_id', team_id).single().execute
        )
        
        is_founder = False
        if subscription_response.data and subscription_response.data['plan_id'] == 'Founder':
            is_founder = True

        if not is_founder:
            # Display activation button if not a founder
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Welcome, <@{user_id}>! Unlock your exclusive Founder benefits."
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Activate Founder Membership",
                                "emoji": True
                            },
                            "style": "primary",
                            "action_id": "open_license_activation_modal"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Already a Founder? Use `/clarity-activate` to submit your key."
                    }
                }
            ]
        else:
            # Display a welcome message for Founders
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Welcome back, Founder <@{user_id}>! Your exclusive membership is active. How can I assist your project today?"
                    }
                }
            ]

        await client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": blocks
            }
        )
    except Exception as e:
        logger.error(f"Error handling app_home_opened event for user {user_id}: {e}")
        await client.chat_postMessage(channel=user_id, text=f"An internal error occurred loading your App Home. Please try again later.")

@slack_app.action("open_license_activation_modal")
async def open_license_modal(ack, body, client, logger):
    await ack()
    logger.info(f"Received open_license_activation_modal action: {body}")

    await client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "activate_license_modal",
            "title": {
                "type": "plain_text",
                "text": "Activate Founder Membership"
            },
            "submit": {
                "type": "plain_text",
                "text": "Activate"
            },
            "blocks": [
                {
                    "type": "input",
                    "block_id": "license_key_input_block",
                    "label": {
                        "type": "plain_text",
                        "text": "Enter your Founder License Key"
                    },
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "license_key_input",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "e.g., CLARITY-FOUNDER-a1b2c3d4-..."
                        },
                        "min_length": 10 # Basic validation
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Your unique key was sent to your email after purchase."
                        }
                    ]
                }
            ]
        }
    )

@slack_app.command("/clarity-activate")
async def clarity_activate_command(ack, body, client, logger):
    await ack()
    logger.info(f"Received /clarity-activate command: {body}")

    await client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "activate_license_modal",
            "title": {
                "type": "plain_text",
                "text": "Activate Founder Membership"
            },
            "submit": {
                "type": "plain_text",
                "text": "Activate"
            },
            "blocks": [
                {
                    "type": "input",
                    "block_id": "license_key_input_block",
                    "label": {
                        "type": "plain_text",
                        "text": "Enter your Founder License Key"
                    },
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "license_key_input",
                        "placeholder": {
                            "type": "plain_text",
                            "text": "e.g., CLARITY-FOUNDER-a1b2c3d4-..."
                        },
                        "min_length": 10
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Your unique key was sent to your email after purchase."
                        }
                    ]
                }
            ]
        }
    )

@slack_app.command("/bot-grant-access")
async def grant_access(ack, body, say, logger, client):
    await ack()
    logger.info(f"Received /bot-grant-access command: {body}")

    user_text_to_grant = body.get("text", "").strip()
    requesting_user_id = body["user_id"]
    team_id = body["team_id"]
    channel_id = body["channel_id"]

    if not await is_user_admin(requesting_user_id, client):
        await say("This command can only be used by workspace admins.")
        return

    if not user_text_to_grant:
        await say("Please provide a user. Usage: `/bot-grant-access @user`")
        return

    parsed_user_id = None
    if user_text_to_grant.startswith('<@U'):
        parsed_user_id = user_text_to_grant.split('|')[0].strip('<@>')
    elif user_text_to_grant.startswith('@'):
        username = user_text_to_grant.strip('@')
        parsed_user_id = await find_user_id_by_name(username, client)
        if not parsed_user_id:
            await say(f"Could not find a user with the name `{username}`.")
            return
    else:
        await say("Please provide a valid user mention or username (e.g., `@user`).")
        return

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

        await asyncio.to_thread(
            rls_supabase_client.from_('authorized_users').insert({
                'workspace_id': team_id,
                'user_id': parsed_user_id
            }).execute
        )
        await say(f"Access granted to <@{parsed_user_id}>.")
    except Exception as e:
        if "violates unique constraint" in str(e):
            await say(f"<@{parsed_user_id}> is already an authorized user.")
        else:
            logger.error(f"Error granting access: {e}")
            await say(f"An error occurred while granting access: {e}")

@slack_app.command("/bot-revoke-access")
async def revoke_access(ack, body, say, logger, client):
    await ack()
    logger.info(f"Received /bot-revoke-access command: {body}")

    user_text_to_revoke = body.get("text", "").strip()
    requesting_user_id = body["user_id"]
    team_id = body["team_id"]
    channel_id = body["channel_id"]

    if not await is_user_admin(requesting_user_id, client):
        await say("This command can only be used by workspace admins.")
        return

    if not user_text_to_revoke:
        await say("Please provide a user. Usage: `/bot-revoke-access @user`")
        return

    parsed_user_id = None
    if user_text_to_revoke.startswith('<@U'):
        parsed_user_id = user_text_to_revoke.split('|')[0].strip('<@>')
    elif user_text_to_revoke.startswith('@'):
        username = user_text_to_revoke.strip('@')
        parsed_user_id = await find_user_id_by_name(username, client)
        if not parsed_user_id:
            await say(f"Could not find a user with the name `{username}`.")
            return
    else:
        await say("Please provide a valid user mention or username (e.g., `@user`).")
        return

    try:
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

        response = await asyncio.to_thread(
            rls_supabase_client.from_('authorized_users')
            .delete()
            .eq('workspace_id', team_id)
            .eq('user_id', parsed_user_id)
            .execute
        )
        
        if response.data:
            await say(f"Access revoked for <@{parsed_user_id}>.")
        else:
            await say(f"<@{parsed_user_id}> was not found in the authorized users list.")

    except Exception as e:
        logger.error(f"Error revoking access: {e}")
        await say(f"An error occurred while revoking access: {e}")

@slack_app.command("/bot-list-authorized")
async def list_authorized_users(ack, body, say, logger, client, context): # Add client and context
    await ack()
    logger.info(f"Received /bot-list-authorized command: {body}")

    user_id = body["user_id"]
    team_id = body["team_id"]

    # First, check if the user is an admin
    user_is_admin = await is_user_admin(user_id, client)
    if not user_is_admin:
        await say("This command can only be used by workspace admins.")
        return

    # If the user is an admin, proceed without the standard authorization check
    channel_id = body["channel_id"]

    try:
        # Create the RLS Supabase client since we bypassed the main auth check
        options = ClientOptions(headers={"x-workspace-id": team_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(
            supabase_url, 
            supabase_service_role_key,
            options=options
        )

        response = await asyncio.to_thread(
            rls_supabase_client.from_('authorized_users').select('user_id').eq('workspace_id', team_id).execute
        )
        authorized_users = response.data

        if authorized_users:
            user_ids = [user['user_id'] for user in authorized_users]
            user_mentions = [f"<@{user_id}>" for user_id in user_ids]
            message = f"Authorized users for this workspace: {', '.join(user_mentions)}"
        else:
            message = "No authorized users found for this workspace."

        await slack_app.client.chat_postMessage(channel=channel_id, text=message) # Use chat_postMessage

    except Exception as e:
        logger.error(f"Error handling /bot-list-authorized command: {e}")
        await slack_app.client.chat_postMessage(channel=channel_id, text=f"An error occurred while listing authorized users: {e}") # Use chat_postMessage
