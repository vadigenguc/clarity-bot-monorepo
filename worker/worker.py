import os
import json
import asyncio
import io
import httpx
import logging
from dotenv import load_dotenv

from pydub import AudioSegment
from pydub.silence import split_on_silence

# Document parsing libraries
from pdfminer.high_level import extract_text as extract_text_from_pdf
from docx import Document
import openpyxl
import pandas as pd

# Vectorization
from sentence_transformers import SentenceTransformer

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

# Load Sentence Transformer model globally
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Worker: SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    logger.error(f"Worker: Error loading SentenceTransformer model: {e}")
    embedding_model = None # Handle case where model fails to load

def get_embedding(text: str):
    if embedding_model:
        return embedding_model.encode(text).tolist()
    logger.error("Worker: Embedding model not loaded. Cannot generate embedding.")
    return None

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
        else:
            logger.warning(f"Worker: Unsupported file type for text extraction: {file_type}")
    except Exception as e:
        logger.error(f"Worker: Error extracting text from {file_type} file: {e}")
    return text

async def process_and_store_embedding(
    workspace_id: str, 
    channel_id: str, 
    source_type: str, 
    source_id: str, 
    content_text: str, 
    rls_supabase_client: Client
):
    """Generates embeddings for content chunks and stores them in Supabase."""
    if not content_text:
        logger.info(f"Worker: No content text to process for {source_type} {source_id} in channel {channel_id}.")
        return

    chunks = chunk_text(content_text)
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            try:
                await asyncio.to_thread(
                    rls_supabase_client.from_('document_embeddings').insert({
                        'workspace_id': workspace_id,
                        'channel_id': channel_id,
                        'source_type': source_type,
                        'source_id': f"{source_id}_chunk_{i}",
                        'content': chunk,
                        'embedding': embedding
                    }).execute
                )
                logger.info(f"Worker: Chunk {i} for {source_type} {source_id} in channel {channel_id} stored in document_embeddings.")
            except Exception as e:
                logger.error(f"Worker: Error storing embedding for chunk {i} of {source_type} {source_id} in channel {channel_id}: {e}")
        else:
            logger.warning(f"Worker: Could not generate embedding for chunk {i} of {source_type} {source_id} in channel {channel_id}.")


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


async def process_embedding_job(job_payload: dict):
    """Processes a single embedding job from the queue."""
    workspace_id = job_payload.get("workspace_id")
    channel_id = job_payload.get("channel_id")
    source_type = job_payload.get("source_type")
    source_id = job_payload.get("source_id")
    content_text = job_payload.get("content")

    if not all([workspace_id, channel_id, source_type, source_id, content_text]):
        logger.error(f"Worker: Invalid embedding job payload: {job_payload}")
        return

    try:
        options = ClientOptions(headers={"x-workspace-id": workspace_id, "x-channel-id": channel_id})
        rls_supabase_client = create_client(supabase_url, supabase_service_role_key, options=options)
        await process_and_store_embedding(workspace_id, channel_id, source_type, source_id, content_text, rls_supabase_client)
        logger.info(f"Worker: Embedding job processed for {source_type} {source_id}.")
    except Exception as e:
        logger.error(f"Worker: Error processing embedding job for {source_type} {source_id}: {e}")

async def process_document_processing_job(job_payload: dict):
    """Processes a single document processing job from the queue."""
    file_url = job_payload.get("file_url")
    workspace_id = job_payload.get("workspace_id")
    channel_id = job_payload.get("channel_id")
    file_id = job_payload.get("file_id")
    file_name = job_payload.get("file_name")
    user_id = job_payload.get("user_id")
    file_type = job_payload.get("file_type")

    if not all([file_url, workspace_id, channel_id, file_id, file_name, user_id, file_type]):
        logger.error(f"Worker: Invalid document processing job payload: {job_payload}")
        return

    slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
    if not slack_bot_token:
        logger.error("SLACK_BOT_TOKEN not set in environment.")
        await send_slack_message(channel_id, user_id, f"❌ Document processing failed for `{file_name}`: Server configuration error.", slack_bot_token)
        return

    try:
        logger.info(f"Worker: Downloading document {file_name} from Slack...")
        file_content = await download_file_from_slack(file_url, slack_bot_token)
        logger.info(f"Worker: Document {file_name} downloaded. Size: {len(file_content) / (1024*1024):.2f} MB")

        logger.info(f"Worker: Extracting text from document {file_name}...")
        extracted_text = extract_text_from_file(file_content, file_type)

        if extracted_text:
            logger.info(f"Worker: Text extracted from {file_name}. Publishing embedding job...")
            embedding_job_payload = {
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "source_type": 'file', # Always 'file' for document processing
                "source_id": file_id,
                "content": extracted_text
            }
            
            publisher, topic_path = get_pubsub_embedding_publisher_client()
            if publisher and topic_path:
                future = publisher.publish(topic_path, json.dumps(embedding_job_payload).encode("utf-8"))
                future.result() # Wait for the publish call to complete
                logger.info(f"Worker: Published embedding job for document {file_name} to Pub/Sub topic: {topic_path}")
                await send_slack_message(channel_id, user_id, f"✅ I've processed `{file_name}` and added it to the project knowledge base (embeddings are being generated).", slack_bot_token)
            else:
                logger.error("Worker: Pub/Sub embedding topic path not configured. Cannot offload embedding for document.")
                await send_slack_message(channel_id, user_id, f"❌ Failed to process document `{file_name}`: Server configuration error (Pub/Sub not set up for embeddings).", slack_bot_token)
        else:
            logger.warning(f"Worker: No text extracted from {file_name}.")
            await send_slack_message(channel_id, user_id, f"⚠️ Could not extract text from `{file_name}`. It might be an unsupported format or empty.", slack_bot_token)

    except Exception as e:
        logger.error(f"Worker: Error processing document job for {file_name}: {e}")
        await send_slack_message(channel_id, user_id, f"❌ An error occurred during document processing for `{file_name}`: {e}", slack_bot_token)

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
            logger.info(f"Worker: Full transcript generated for {file_name}. Publishing embedding job...")
            embedding_job_payload = {
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "source_type": 'transcription',
                "source_id": file_id,
                "content": full_transcript
            }
            publisher, topic_path = get_pubsub_embedding_publisher_client()
            if publisher and topic_path:
                future = publisher.publish(topic_path, json.dumps(embedding_job_payload).encode("utf-8"))
                future.result() # Wait for the publish call to complete
                logger.info(f"Worker: Published embedding job for transcription of {file_name} to Pub/Sub topic: {topic_path}")
            else:
                logger.error("Worker: Pub/Sub embedding topic path not configured. Cannot offload embedding for transcription.")

            # Load summarization prompt
            summarization_prompt = load_prompt("summarization_prompt")
            logger.info(f"Worker: Generating summary for {file_name}...")
            summary = await llm_service_manager.summarize_text(full_transcript, summarization_prompt)

            if summary:
                logger.info(f"Worker: Summary generated for {file_name}. Publishing embedding job for summary...")
                summary_embedding_job_payload = {
                    "workspace_id": workspace_id,
                    "channel_id": channel_id,
                    "source_type": 'summary',
                    "source_id": file_id,
                    "content": summary
                }
                publisher, topic_path = get_pubsub_embedding_publisher_client()
                if publisher and topic_path:
                    future = publisher.publish(topic_path, json.dumps(summary_embedding_job_payload).encode("utf-8"))
                    future.result() # Wait for the publish call to complete
                    logger.info(f"Worker: Published embedding job for summary of {file_name} to Pub/Sub topic: {topic_path}")
                else:
                    logger.error("Worker: Pub/Sub embedding topic path not configured. Cannot offload embedding for summary.")
                
                # Send initial message about transcription completion
                await send_slack_message(channel_id, user_id, f"✅ Your transcription for `{file_name}` is complete! It has been added to the project knowledge base (embeddings are being generated).", slack_bot_token)
                
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
PUBSUB_TRANSCRIPTION_SUBSCRIPTION_NAME = "clarity-transcription-jobs-sub" # Name of the Pub/Sub subscription for transcription
PUBSUB_EMBEDDING_SUBSCRIPTION_NAME = "clarity-embedding-jobs-sub" # Name of the Pub/Sub subscription for embeddings
PUBSUB_DOCUMENT_PROCESSING_SUBSCRIPTION_NAME = "clarity-document-processing-jobs-sub" # Name of the Pub/Sub subscription for document processing
PUBSUB_MESSAGE_SUBSCRIPTION_NAME = "clarity-message-jobs-sub" # Name of the Pub/Sub subscription for messages

# Lazy initialization for Pub/Sub subscriber clients
_pubsub_transcription_subscriber = None
_pubsub_transcription_subscription_path = None
_pubsub_embedding_subscriber = None
_pubsub_embedding_subscription_path = None
_pubsub_document_processing_subscriber = None
_pubsub_document_processing_subscription_path = None
_pubsub_message_subscriber = None
_pubsub_message_subscription_path = None

def get_pubsub_transcription_subscriber_client():
    global _pubsub_transcription_subscriber, _pubsub_transcription_subscription_path
    if _pubsub_transcription_subscriber is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub transcription subscriber will not function.")
            return None, None
        _pubsub_transcription_subscriber = pubsub_v1.SubscriberClient()
        _pubsub_transcription_subscription_path = _pubsub_transcription_subscriber.subscription_path(GCP_PROJECT_ID, PUBSUB_TRANSCRIPTION_SUBSCRIPTION_NAME)
        logger.info("Pub/Sub transcription subscriber client initialized.")
    return _pubsub_transcription_subscriber, _pubsub_transcription_subscription_path

def get_pubsub_embedding_subscriber_client():
    global _pubsub_embedding_subscriber, _pubsub_embedding_subscription_path
    if _pubsub_embedding_subscriber is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub embedding subscriber will not function.")
            return None, None
        _pubsub_embedding_subscriber = pubsub_v1.SubscriberClient()
        _pubsub_embedding_subscription_path = _pubsub_embedding_subscriber.subscription_path(GCP_PROJECT_ID, PUBSUB_EMBEDDING_SUBSCRIPTION_NAME)
        logger.info("Pub/Sub embedding subscriber client initialized.")
    return _pubsub_embedding_subscriber, _pubsub_embedding_subscription_path

def get_pubsub_document_processing_subscriber_client():
    global _pubsub_document_processing_subscriber, _pubsub_document_processing_subscription_path
    if _pubsub_document_processing_subscriber is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub document processing subscriber will not function.")
            return None, None
        _pubsub_document_processing_subscriber = pubsub_v1.SubscriberClient()
        _pubsub_document_processing_subscription_path = _pubsub_document_processing_subscriber.subscription_path(GCP_PROJECT_ID, PUBSUB_DOCUMENT_PROCESSING_SUBSCRIPTION_NAME)
        logger.info("Pub/Sub document processing subscriber client initialized.")
    return _pubsub_document_processing_subscriber, _pubsub_document_processing_subscription_path

def get_pubsub_message_subscriber_client():
    global _pubsub_message_subscriber, _pubsub_message_subscription_path
    if _pubsub_message_subscriber is None:
        if not GCP_PROJECT_ID:
            logger.error("GCP_PROJECT_ID environment variable not set. Pub/Sub message subscriber will not function.")
            return None, None
        _pubsub_message_subscriber = pubsub_v1.SubscriberClient()
        _pubsub_message_subscription_path = _pubsub_message_subscriber.subscription_path(GCP_PROJECT_ID, PUBSUB_MESSAGE_SUBSCRIPTION_NAME)
        logger.info("Pub/Sub message subscriber client initialized.")
    return _pubsub_message_subscriber, _pubsub_message_subscription_path

async def process_message_job(job_payload: dict):
    """Processes a single message job from the queue."""
    # This function will contain the logic from the old process_message_background
    # For now, we'll just log it. A full implementation would require the authorization logic as well.
    logger.info(f"Worker: Processing message job: {job_payload.get('text')}")
    # The full logic from backend/main.py's process_message_background would go here.
    # This includes authorization checks, storing the message, and sending a reply.
    # For this example, we'll just log that we received it.
    pass

async def main_worker_loop():
    """Main loop for the Cloud Run Job to pull and process messages from all subscriptions."""
    
    # Initialize all subscribers and their paths
    transcription_subscriber, transcription_sub_path = get_pubsub_transcription_subscriber_client()
    embedding_subscriber, embedding_sub_path = get_pubsub_embedding_subscriber_client()
    document_processing_subscriber, document_processing_sub_path = get_pubsub_document_processing_subscriber_client()
    message_subscriber, message_sub_path = get_pubsub_message_subscriber_client()

    subscribers_info = []
    if transcription_subscriber and transcription_sub_path:
        subscribers_info.append({"subscriber": transcription_subscriber, "path": transcription_sub_path, "type": "transcription"})
    if embedding_subscriber and embedding_sub_path:
        subscribers_info.append({"subscriber": embedding_subscriber, "path": embedding_sub_path, "type": "embedding"})
    if document_processing_subscriber and document_processing_sub_path:
        subscribers_info.append({"subscriber": document_processing_subscriber, "path": document_processing_sub_path, "type": "document_processing"})
    if message_subscriber and message_sub_path:
        subscribers_info.append({"subscriber": message_subscriber, "path": message_sub_path, "type": "message"})

    if not subscribers_info:
        logger.error("Worker: No Pub/Sub subscribers or subscription paths configured. Exiting worker.")
        return

    logger.info(f"Worker: Pulling messages from {len(subscribers_info)} subscriptions.")
    
    all_ack_ids = []
    all_tasks = []

    for sub_info in subscribers_info:
        subscriber = sub_info["subscriber"]
        sub_path = sub_info["path"]
        sub_type = sub_info["type"]

        logger.info(f"Worker: Pulling messages from {sub_type} subscription: {sub_path}")
        try:
            response = subscriber.pull(
                request={
                    "subscription": sub_path,
                    "max_messages": 10, # Process up to 10 messages per job run per subscription
                    "return_immediately": True,
                }
            )

            if not response.received_messages:
                logger.info(f"Worker: No messages to process in {sub_type} subscription.")
                continue

            for received_message in response.received_messages:
                all_ack_ids.append({"ack_id": received_message.ack_id, "subscriber": subscriber, "sub_path": sub_path})
                try:
                    job_payload = json.loads(received_message.message.data.decode("utf-8"))
                    logger.info(f"Worker: Received {sub_type} job: {job_payload.get('file_name', job_payload.get('source_id', 'N/A'))}")
                    
                    if sub_type == "transcription":
                        all_tasks.append(process_transcription_job(job_payload))
                    elif sub_type == "embedding":
                        all_tasks.append(process_embedding_job(job_payload))
                    elif sub_type == "document_processing":
                        all_tasks.append(process_document_processing_job(job_payload))
                    elif sub_type == "message":
                        all_tasks.append(process_message_job(job_payload))
                    else:
                        logger.warning(f"Worker: Unhandled job type '{sub_type}' from subscription {sub_path}. Acknowledging message.")
                        # Acknowledge unhandled messages to remove them from the queue
                        subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": [received_message.ack_id]})

                except json.JSONDecodeError as e:
                    logger.error(f"Worker: Failed to decode Pub/Sub message from {sub_type} subscription: {e}. Message: {received_message.message.data}")
                    # Acknowledge malformed messages
                    subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": [received_message.ack_id]})
                
        except Exception as e:
            logger.error(f"Worker: An error occurred during Pub/Sub message pulling from {sub_type} subscription {sub_path}: {e}")

    if all_tasks:
        await asyncio.gather(*all_tasks) # Process all valid jobs concurrently

    # Acknowledge all processed messages for each subscriber
    for sub_info in subscribers_info:
        subscriber = sub_info["subscriber"]
        sub_path = sub_info["path"]
        ack_ids_for_sub = [item["ack_id"] for item in all_ack_ids if item["subscriber"] == subscriber]
        if ack_ids_for_sub:
            try:
                subscriber.acknowledge(request={"subscription": sub_path, "ack_ids": ack_ids_for_sub})
                logger.info(f"Worker: Acknowledged {len(ack_ids_for_sub)} messages for subscription {sub_path}.")
            except Exception as e:
                logger.error(f"Worker: Error acknowledging messages for subscription {sub_path}: {e}")

# --- Main execution for Google Cloud Run Job ---
if __name__ == "__main__":
    logger.info("Cloud Run Job Worker started.")
    # Ensure GCP_PROJECT_ID is set in the environment for the worker
    if not os.environ.get("GCP_PROJECT_ID"):
        logger.error("GCP_PROJECT_ID environment variable not set. Exiting.")
        exit(1)
    asyncio.run(main_worker_loop())
    logger.info("Cloud Run Job Worker finished.")
