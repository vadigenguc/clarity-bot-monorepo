import pytest
import os
import io
from unittest.mock import patch, AsyncMock, MagicMock
from openai import AsyncOpenAI
from vertexai.preview.generative_models import GenerativeModel, Part
from backend.services.llm_service import LLMServiceManager

# Fixture to mock the underlying client classes and aiplatform.init
@pytest.fixture(autouse=True, scope="function")
def mock_llm_client_classes():
    with patch('backend.services.llm_service.AsyncOpenAI') as mock_openai_class, \
         patch('backend.services.llm_service.GenerativeModel') as mock_gemini_class, \
         patch('backend.services.llm_service.aiplatform.init', MagicMock()) as mock_aiplatform_init_patch:
        
        mock_openai_instance = AsyncMock(spec=AsyncOpenAI)
        mock_openai_class.return_value = mock_openai_instance
        
        mock_gemini_instance = MagicMock(spec=GenerativeModel)
        mock_gemini_class.return_value = mock_gemini_instance

        yield mock_openai_class, mock_gemini_class, mock_openai_instance, mock_gemini_instance, mock_aiplatform_init_patch

# --- Test LLMServiceManager Initialization ---

@pytest.mark.asyncio
async def test_openai_client_initialization_success(mock_llm_client_classes):
    mock_openai_class, _, _, _, _ = mock_llm_client_classes
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        manager = LLMServiceManager()
        mock_openai_class.assert_called_once_with(api_key="test_key")
        assert manager._openai_client is not None

@pytest.mark.asyncio
async def test_openai_client_initialization_no_key(mock_llm_client_classes):
    mock_openai_class, _, _, _, _ = mock_llm_client_classes
    with patch.dict(os.environ, {}, clear=True): # Clear OPENAI_API_KEY
        manager = LLMServiceManager()
        mock_openai_class.assert_not_called()
        assert manager._openai_client is None

@pytest.mark.asyncio
async def test_gemini_client_initialization_success(mock_llm_client_classes):
    _, mock_gemini_class, _, _, mock_aiplatform_init = mock_llm_client_classes
    with patch.dict(os.environ, {"GCP_PROJECT_ID": "test_project", "GCP_LOCATION": "test_location"}):
        manager = LLMServiceManager()
        mock_aiplatform_init.assert_called_once_with(project="test_project", location="test_location")
        mock_gemini_class.assert_called_once_with("gemini-pro")
        assert manager._gemini_client is not None

@pytest.mark.asyncio
async def test_gemini_client_initialization_no_project_id(mock_llm_client_classes):
    _, mock_gemini_class, _, _, mock_aiplatform_init = mock_llm_client_classes
    with patch.dict(os.environ, {"GCP_LOCATION": "test_location"}, clear=True): # Clear GCP_PROJECT_ID
        manager = LLMServiceManager()
        mock_aiplatform_init.assert_not_called()
        mock_gemini_class.assert_not_called()
        assert manager._gemini_client is None

# --- Test generate_text method ---

@pytest.mark.asyncio
async def test_generate_text_openai_chat_success(mock_llm_client_classes):
    _, _, mock_openai_instance, _, _ = mock_llm_client_classes
    mock_openai_instance.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="OpenAI response"))])
    )
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("openai-gpt-4", "Hello")
        assert result == "OpenAI response"
        mock_openai_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[{"role": "user", "content": "Hello"}]
        )

@pytest.mark.asyncio
async def test_generate_text_openai_chat_failure(mock_llm_client_classes):
    _, _, mock_openai_instance, _, _ = mock_llm_client_classes
    mock_openai_instance.chat.completions.create.side_effect = Exception("OpenAI API error")
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("openai-gpt-4", "Hello")
        assert result is None
        mock_openai_instance.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_openai_whisper_success(mock_llm_client_classes):
    _, _, mock_openai_instance, _, _ = mock_llm_client_classes
    mock_create_method = AsyncMock(return_value=MagicMock(text="Whisper transcription"))
    mock_openai_instance.audio.transcriptions.create = mock_create_method
    
    dummy_audio_file = MagicMock(spec=io.BytesIO)
    dummy_audio_file.name = "audio.mp3"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("openai-whisper-1", dummy_audio_file)
        assert result == "Whisper transcription"
        mock_openai_instance.audio.transcriptions.create.assert_called_once_with(
            model="whisper-1", file=dummy_audio_file
        )

@pytest.mark.asyncio
async def test_generate_text_openai_whisper_failure(mock_llm_client_classes):
    _, _, mock_openai_instance, _, _ = mock_llm_client_classes
    mock_openai_instance.audio.transcriptions.create.side_effect = Exception("Whisper API error")
    
    dummy_audio_file = MagicMock(spec=io.BytesIO)
    dummy_audio_file.name = "audio.mp3"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy_key"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("openai-whisper-1", dummy_audio_file)
        assert result is None
        mock_openai_instance.audio.transcriptions.create.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_gemini_success(mock_llm_client_classes):
    _, _, _, mock_gemini_instance, _ = mock_llm_client_classes
    mock_gemini_instance.generate_content_async.return_value = AsyncMock(candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text="Gemini response")]))])
    
    with patch.dict(os.environ, {"GCP_PROJECT_ID": "dummy_project"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("gemini-pro", "Hello")
        assert result == "Gemini response"
        mock_gemini_instance.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_gemini_failure(mock_llm_client_classes):
    _, _, _, mock_gemini_instance, _ = mock_llm_client_classes
    mock_gemini_instance.generate_content_async.side_effect = Exception("Gemini API error")
    
    with patch.dict(os.environ, {"GCP_PROJECT_ID": "dummy_project"}):
        manager = LLMServiceManager()
        result = await manager.generate_text("gemini-pro", "Hello")
        assert result is None
        mock_gemini_instance.generate_content_async.assert_called_once()

@pytest.mark.asyncio
async def test_generate_text_unsupported_model(mock_llm_client_classes):
    # No need to initialize clients for unsupported model test
    manager = LLMServiceManager()
    result = await manager.generate_text("unsupported-model", "Hello")
    assert result is None

# --- Test summarize_text method ---

@pytest.mark.asyncio
async def test_summarize_text_success(mock_llm_client_classes):
    _, _, _, _, _ = mock_llm_client_classes
    
    # Mock the generate_text method directly on the manager instance
    with patch('backend.services.llm_service.LLMServiceManager.generate_text', new_callable=AsyncMock) as mock_generate_text:
        mock_generate_text.return_value = "This is a summary."
        
        manager = LLMServiceManager()
        text_to_summarize = "Long text content that needs to be summarized."
        summarization_prompt = "Please summarize the following text:"
        
        result = await manager.summarize_text(text_to_summarize, summarization_prompt)
        
        expected_full_prompt = f"{summarization_prompt}\n\nText to summarize:\n{text_to_summarize}"
        
        mock_generate_text.assert_called_once_with(
            model_name="gemini-pro", 
            prompt=expected_full_prompt
        )
        assert result == "This is a summary."

@pytest.mark.asyncio
async def test_summarize_text_failure(mock_llm_client_classes):
    _, _, _, _, _ = mock_llm_client_classes
    
    with patch('backend.services.llm_service.LLMServiceManager.generate_text', new_callable=AsyncMock) as mock_generate_text:
        mock_generate_text.return_value = None # Simulate failure in generate_text
        
        manager = LLMServiceManager()
        text_to_summarize = "Long text content that needs to be summarized."
        summarization_prompt = "Please summarize the following text:"
        
        result = await manager.summarize_text(text_to_summarize, summarization_prompt)
        
        assert result is None
        mock_generate_text.assert_called_once()
