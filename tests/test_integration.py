import pytest
import os
import asyncio
import io # Import io
from unittest.mock import patch, AsyncMock, MagicMock, ANY
from fastapi.testclient import TestClient
from backend.main import app, transcribe_audio
from worker.worker import transcribe_audio_chunk
from backend.services.llm_service import llm_service_manager
from backend.utils.prompt_loader import load_prompt
from supabase import create_client, Client, ClientOptions

@pytest.fixture(scope="module")
def test_app():
    # Import app and other global objects here, after global mocks (from conftest.py) are applied
    # This ensures that global initializations in backend.main happen AFTER mocks are active.
    # from backend.main import app, slack_app, supabase, llm_service_manager
    # from backend.services.llm_service import LLMServiceManager
    # from backend.utils.prompt_loader import load_prompt

    # Use a TestClient for FastAPI application
    client = TestClient(app)
    yield client

@pytest.mark.asyncio
async def test_transcribe_audio_integration():
    """Test that transcribe_audio in main.py calls the LLMServiceManager."""
    with patch('backend.main.llm_service_manager.generate_text', new_callable=AsyncMock) as mock_generate_text:
        mock_generate_text.return_value = "test transcription"
        result = await transcribe_audio(b"test_audio_content", "test_file.mp3")
        assert result == "test transcription"
        mock_generate_text.assert_called_once()

@pytest.mark.asyncio
async def test_transcribe_audio_chunk_integration():
    """Test that transcribe_audio_chunk in worker.py calls the LLMServiceManager."""
    with patch('worker.worker.llm_service_manager.generate_text', new_callable=AsyncMock) as mock_generate_text:
        mock_generate_text.return_value = "test transcription"
        
        # Create a proper file-like object with a name attribute for the test
        dummy_audio_file_io = MagicMock(spec=io.BytesIO)
        dummy_audio_file_io.name = "test_file.mp3"

        result = await transcribe_audio_chunk(dummy_audio_file_io, "test_file.mp3")
        assert result == "test transcription"
        mock_generate_text.assert_called_once_with(
            model_name="openai-whisper-1", 
            prompt=dummy_audio_file_io
        )

@pytest.mark.asyncio
async def test_process_transcription_job_with_summarization_integration():
    """Test the full transcription and summarization flow in worker.py."""
    job_payload = {
        "file_url": "http://example.com/audio.mp3",
        "workspace_id": "T12345",
        "channel_id": "C12345",
        "file_id": "F12345",
        "file_name": "audio.mp3",
        "user_id": "U12345",
        "file_type": "mp3"
    }
    
    mock_audio_content = b"dummy_audio_data"
    mock_transcript = "This is a test transcription."
    mock_summary = "This is a test summary."
    mock_summarization_prompt_content = "Please summarize the following text:"

    with patch('worker.worker.download_file_from_slack', new_callable=AsyncMock) as mock_download_file, \
         patch('worker.worker.split_audio_into_chunks', new_callable=AsyncMock) as mock_split_audio, \
         patch('worker.worker.transcribe_audio_chunk', new_callable=AsyncMock) as mock_transcribe_chunk, \
         patch('worker.worker.llm_service_manager.summarize_text', new_callable=AsyncMock) as mock_summarize_text, \
         patch('worker.worker.process_and_store_content', new_callable=AsyncMock) as mock_process_and_store, \
         patch('worker.worker.send_slack_message', new_callable=AsyncMock) as mock_send_slack_message, \
         patch('worker.worker.load_prompt') as mock_load_prompt:
        
        mock_download_file.return_value = mock_audio_content
        mock_split_audio.return_value = [io.BytesIO(b"chunk1"), io.BytesIO(b"chunk2")]
        mock_transcribe_chunk.return_value = mock_transcript
        mock_summarize_text.return_value = mock_summary
        mock_load_prompt.return_value = mock_summarization_prompt_content

        # Set environment variable for SLACK_BOT_TOKEN
        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
            from worker.worker import process_transcription_job # Re-import to pick up mocks

            await process_transcription_job(job_payload)

            mock_download_file.assert_called_once_with(job_payload["file_url"], "xoxb-test-token")
            mock_split_audio.assert_called_once_with(mock_audio_content, job_payload["file_type"])
            assert mock_transcribe_chunk.call_count == 2 # Called for each chunk
            mock_load_prompt.assert_called_once_with("summarization_prompt")
            mock_summarize_text.assert_called_once_with(
                f"{mock_transcript} {mock_transcript}", # Full transcript from two chunks
                mock_summarization_prompt_content
            )
            
            # Assert process_and_store_content was called for transcription and summary
            assert mock_process_and_store.call_count == 2
            mock_process_and_store.assert_any_call(
                job_payload["workspace_id"],
                job_payload["channel_id"],
                'transcription',
                job_payload["file_id"],
                f"{mock_transcript} {mock_transcript}",
                ANY # Match any object for the rls_supabase_client
            )
            mock_process_and_store.assert_any_call(
                job_payload["workspace_id"],
                job_payload["channel_id"],
                'summary',
                job_payload["file_id"],
                mock_summary,
                ANY # Match any object for the rls_supabase_client
            )

            # Assert Slack messages were sent
            assert mock_send_slack_message.call_count == 2
            mock_send_slack_message.assert_any_call(
                job_payload["channel_id"],
                job_payload["user_id"],
                f"✅ Your transcription for `{job_payload['file_name']}` is complete! It has been added to the project knowledge base.",
                "xoxb-test-token"
            )
            mock_send_slack_message.assert_any_call(
                job_payload["channel_id"],
                job_payload["user_id"],
                f"📝 Here's a summary of `{job_payload['file_name']}`:\n\n{mock_summary}",
                "xoxb-test-token"
            )
