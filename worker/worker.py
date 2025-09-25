import os
import json
import asyncio
import io
import httpx
import logging
from dotenv import load_dotenv

from pydub import AudioSegment
from pydub.silence import split_on_silence

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

async def download_file_from_slack(file_url: str, slack_bot_token: str) -> bytes:
    """Downloads a file from Slack."""
    headers = {'Authorization': f'Bearer {slack_bot_token}'}
    async with httpx.AsyncClient() as client:
        response = await client.get(file_url, headers=headers)
        response.raise_for_status()
        return response.content

async def split_audio_into_chunks(audio_content: bytes, file_type: str) -> list[io.BytesIO]:
    """Splits audio content into chunks suitable for Whisper API."""
    audio = AudioSegment.from_file(io.BytesIO(audio_content), format=file_type)
    
    # Split on silence to get natural segments, then re-chunk if segments are too large
    # Adjust silence_thresh and min_silence_len as needed for your audio
    segments = split_on_silence(
        audio,
        min_silence_len=1000,  # milliseconds of silence
        silence_thresh=-40,    # dBFS below which is considered silence
        keep_silence=200       # keep 200ms of silence at the beginning and end of segments
    )

    chunks = []
    current_chunk = AudioSegment.empty()
    for segment in segments:
        # Estimate size of current_chunk + segment. If it exceeds WHISPER_MAX_FILE_SIZE_BYTES,
        # start a new chunk. This is an approximation, actual size depends on encoding.
        # A more robust solution might involve actual byte size estimation or a fixed duration.
        if len(current_chunk) > 0 and (len(current_chunk) + len(segment)) * (audio.frame_rate / 1000) * (audio.sample_width * audio.channels) > WHISPER_MAX_FILE_SIZE_BYTES:
            chunk_io = io.BytesIO()
            current_chunk.export(chunk_io, format="mp3") # Export as mp3 for Whisper
            chunk_io.seek(0)
            chunks.append(chunk_io)
            current_chunk = segment
        else:
            current_chunk += segment
    
    if len(current_chunk) > 0:
        chunk_io = io.BytesIO()
        current_chunk.export(chunk_io, format="mp3")
        chunk_io.seek(0)
        chunks.append(chunk_io)

    return chunks

async def transcribe_audio_chunk(audio_chunk_io: io.BytesIO, chunk_name: str) -> str | None:
    """Transcribes a single audio chunk using OpenAI Whisper via LLMServiceManager."""
    try:
        # Ensure the file-like object has a 'name' attribute for the LLMServiceManager
        audio_chunk_io.name = chunk_name 
        
        # Use the LLMServiceManager for transcription
        # Assuming 'whisper-1' is an OpenAI model, prefix with 'openai-'
        transcript_text = await llm_service_manager.generate_text(
            model_name="openai-whisper-1", 
            prompt=audio_chunk_io # Pass the file-like object directly
        )
        return transcript_text
    except Exception as e:
        logger.error(f"Error transcribing audio chunk '{chunk_name}': {e}")
        return None

async def process_and_store_content(
    workspace_id: str, 
    channel_id: str, 
    source_type: str, 
    source_id: str, 
    content_text: str, 
    rls_supabase_client: Client
):
    """Stores content and its embeddings in Supabase."""
    if not content_text:
        logger.info(f"No content text to process for {source_type} {source_id} in channel {channel_id}.")
        return

    # Placeholder for chunking and embedding logic (similar to main.py)
    # For a real worker, you'd re-implement or import these functions.
    # For now, we'll just store the full text as one entry.
    try:
        options = ClientOptions(headers={"x-workspace-id": workspace_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)

        # In a real scenario, you'd chunk and embed here.
        # For simplicity, storing the full transcript as one entry for now.
        await asyncio.to_thread(
            rls_supabase_client.from_('document_embeddings').insert({
                'workspace_id': workspace_id,
                'channel_id': channel_id,
                'source_type': source_type,
                'source_id': source_id,
                'content': content_text,
                'embedding': [] # Placeholder for actual embedding
            }).execute
        )
        logger.info(f"Content for {source_type} {source_id} in channel {channel_id} stored in document_embeddings.")
    except Exception as e:
        logger.error(f"Error storing content for {source_type} {source_id} in channel {channel_id}: {e}")


async def send_slack_message(channel_id: str, user_id: str, text: str, slack_bot_token: str):
    """Sends a Slack message to a user or channel."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {slack_bot_token}",
                "Content-Type": "application/json"
            },
            json={
                "channel": channel_id,
                "text": text,
                "user": user_id # To ensure the message is visible to the user in DMs or mentions
            }
        )
        response.raise_for_status()
        logger.info(f"Slack message sent to {channel_id} for user {user_id}.")


async def process_transcription_job(job_payload: dict):
    """Processes a single transcription job from the queue."""
    file_url = job_payload.get("file_url")
    workspace_id = job_payload.get("workspace_id")
    channel_id = job_payload.get("channel_id")
    file_id = job_payload.get("file_id")
    file_name = job_payload.get("file_name")
    user_id = job_payload.get("user_id")
    file_type = job_payload.get("file_type")

    if not all([file_url, workspace_id, channel_id, file_id, file_name, user_id, file_type]):
        logger.error(f"Invalid job payload: {job_payload}")
        return

    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_bot_token:
        logger.error("SLACK_BOT_TOKEN not set in environment.")
        await send_slack_message(channel_id, user_id, f"❌ Transcription failed for `{file_name}`: Server configuration error.", slack_bot_token)
        return

    try:
        logger.info(f"Worker: Downloading file {file_name} from Slack...")
        audio_content = await download_file_from_slack(file_url, slack_bot_token)
        logger.info(f"Worker: File {file_name} downloaded. Size: {len(audio_content) / (1024*1024):.2f} MB")

        logger.info(f"Worker: Splitting audio {file_name} into chunks...")
        audio_chunks_io = await split_audio_into_chunks(audio_content, file_type)
        logger.info(f"Worker: Split {file_name} into {len(audio_chunks_io)} chunks.")

        transcription_tasks = []
        for i, chunk_io in enumerate(audio_chunks_io):
            transcription_tasks.append(transcribe_audio_chunk(chunk_io, f"{file_name}_chunk_{i}.mp3"))
        
        logger.info(f"Worker: Transcribing chunks for {file_name} in parallel...")
        transcripts = await asyncio.gather(*transcription_tasks)
        
        full_transcript = " ".join([t for t in transcripts if t])
        
        if full_transcript:
            logger.info(f"Worker: Full transcript generated for {file_name}. Storing in Supabase...")
            options = ClientOptions(headers={"x-workspace-id": workspace_id, "x-channel-id": channel_id})
            rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)
            await process_and_store_content(workspace_id, channel_id, 'transcription', file_id, full_transcript, rls_supabase_client)
            
            # Load summarization prompt
            summarization_prompt = load_prompt("summarization_prompt")
            logger.info(f"Worker: Generating summary for {file_name}...")
            summary = await llm_service_manager.summarize_text(full_transcript, summarization_prompt)

            if summary:
                logger.info(f"Worker: Summary generated for {file_name}. Storing in Supabase and sending to Slack...")
                await process_and_store_content(workspace_id, channel_id, 'summary', file_id, summary, rls_supabase_client)
                
                # Send initial message about transcription completion
                await send_slack_message(channel_id, user_id, f"✅ Your transcription for `{file_name}` is complete! It has been added to the project knowledge base.", slack_bot_token)
                
                # Send summary as a threaded reply
                await send_slack_message(channel_id, user_id, f"📝 Here's a summary of `{file_name}`:\n\n{summary}", slack_bot_token)
            else:
                logger.warning(f"Worker: No summary generated for {file_name}.")
                await send_slack_message(channel_id, user_id, f"⚠️ Transcription complete for `{file_name}`, but summary generation failed.", slack_bot_token)
        else:
            logger.warning(f"Worker: No transcript generated for {file_name}.")
            await send_slack_message(channel_id, user_id, f"❌ Transcription failed for `{file_name}`: No text could be extracted.", slack_bot_token)

    except Exception as e:
        logger.error(f"Worker: Error processing transcription job for {file_name}: {e}")
        await send_slack_message(channel_id, user_id, f"❌ An error occurred during transcription for `{file_name}`: {e}", slack_bot_token)


GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
PUBSUB_SUBSCRIPTION_NAME = "clarity-transcription-jobs-sub" # Name of the Pub/Sub subscription

def get_pubsub_subscriber_client():
    """Initializes and returns a Pub/Sub subscriber client."""
    if not GCP_PROJECT_ID:
        logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub subscriber will not function.")
        return None
    return pubsub_v1.SubscriberClient()

def get_pubsub_subscription_path(subscriber_client):
    """Returns the full Pub/Sub subscription path."""
    if not GCP_PROJECT_ID or not subscriber_client:
        return None
    return subscriber_client.subscription_path(GCP_PROJECT_ID, PUBSUB_SUBSCRIPTION_NAME)

async def main_worker_loop():
    """Main loop for the Cloud Run Job to pull and process messages."""
    subscriber = get_pubsub_subscriber_client()
    PUBSUB_SUBSCRIPTION_PATH = get_pubsub_subscription_path(subscriber)

    if not subscriber or not PUBSUB_SUBSCRIPTION_PATH:
        logger.error("Pub/Sub subscriber or subscription path not configured. Exiting worker.")
        return

    logger.info(f"Worker: Pulling messages from subscription: {PUBSUB_SUBSCRIPTION_PATH}")
    
    try:
        # Pull messages (blocking call, but Cloud Run Job will run to completion)
        # For a Cloud Run Job, we typically pull a finite number of messages and then exit.
        # The Cloud Scheduler will trigger this job periodically.
        response = subscriber.pull(
            request={
                "subscription": PUBSUB_SUBSCRIPTION_PATH,
                "max_messages": 10, # Process up to 10 messages per job run
                "return_immediately": True, # Return even if no messages
            }
        )

        if not response.received_messages:
            logger.info("Worker: No messages to process. Exiting.")
            return

        ack_ids = []
        tasks = []
        for received_message in response.received_messages:
            ack_ids.append(received_message.ack_id)
            try:
                job_payload = json.loads(received_message.message.data.decode("utf-8"))
                logger.info(f"Worker: Received job: {job_payload.get('file_name', 'N/A')}")
                tasks.append(process_transcription_job(job_payload))
            except json.JSONDecodeError as e:
                logger.error(f"Worker: Failed to decode Pub/Sub message: {e}. Message: {received_message.message.data}")
                # Acknowledge malformed messages to remove them from the queue
                subscriber.acknowledge(request={"subscription": PUBSUB_SUBSCRIPTION_PATH, "ack_ids": [received_message.ack_id]})
                
        if tasks:
            await asyncio.gather(*tasks) # Process all valid jobs concurrently

        if ack_ids:
            # Acknowledge all processed messages
            subscriber.acknowledge(request={"subscription": PUBSUB_SUBSCRIPTION_PATH, "ack_ids": ack_ids})
            logger.info(f"Worker: Acknowledged {len(ack_ids)} messages.")

    except Exception as e:
        logger.error(f"Worker: An error occurred during Pub/Sub message pulling or processing: {e}")

# --- Main execution for Google Cloud Run Job ---
if __name__ == "__main__":
    logger.info("Cloud Run Job Worker started.")
    # Ensure GCP_PROJECT_ID is set in the environment for the worker
    if not os.environ.get("GCP_PROJECT_ID"):
        logger.error("GCP_PROJECT_ID environment variable not set. Exiting.")
        exit(1)
    asyncio.run(main_worker_loop())
    logger.info("Cloud Run Job Worker finished.")
