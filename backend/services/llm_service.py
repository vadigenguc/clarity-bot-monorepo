import os
import logging
from openai import AsyncOpenAI
from google.cloud import aiplatform # Keep this for aiplatform.init
from vertexai.preview.generative_models import GenerativeModel, Part

logger = logging.getLogger(__name__)

class LLMServiceManager:
    _openai_client: AsyncOpenAI | None = None
    _gemini_client: GenerativeModel | None = None

    def __init__(self):
        self._initialize_openai_client()
        self._initialize_gemini_client()

    def _initialize_openai_client(self):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized.")
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI client not initialized.")

    def _initialize_gemini_client(self):
        gcp_project_id = os.environ.get("GCP_PROJECT_ID")
        gcp_location = os.environ.get("GCP_LOCATION", "europe-west1") # Default to europe-west1 as per docs
        
        if gcp_project_id:
            try:
                aiplatform.init(project=gcp_project_id, location=gcp_location)
                self._gemini_client = GenerativeModel("gemini-2.5-flash")
                logger.info(f"Google Gemini client initialized for project {gcp_project_id} in {gcp_location}.")
            except Exception as e:
                logger.error(f"Error initializing Google Gemini client: {e}")
                self._gemini_client = None
        else:
            logger.warning("GCP_PROJECT_ID not found. Google Gemini client not initialized.")

    async def generate_text(self, model_name: str, prompt: str, **kwargs) -> str | None:
        """
        Generates text using the specified LLM model.
        :param model_name: The name of the LLM model to use (e.g., "openai-gpt-4", "gemini-pro").
        :param prompt: The prompt to send to the LLM.
        :param kwargs: Additional parameters for the LLM API call.
        :return: Generated text or None if an error occurs.
        """
        if model_name.startswith("openai-"):
            if not self._openai_client:
                logger.error("OpenAI client not initialized.")
                return None
            try:
                if model_name == "openai-whisper-1":
                    # For Whisper, 'prompt' is expected to be a file-like object
                    response = await self._openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=prompt, # Pass the file-like object directly
                        **kwargs
                    )
                    return response.text
                else:
                    response = await self._openai_client.chat.completions.create(
                        model=model_name.replace("openai-", ""), # e.g., "gpt-4"
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating text with OpenAI model {model_name}: {e}")
                return None
        elif model_name.startswith("gemini-"):
            if not self._gemini_client:
                logger.error("Google Gemini client not initialized.")
                return None
            try:
                # Gemini API expects content in a specific format
                contents = [Part.from_text(prompt)]
                response = await self._gemini_client.generate_content_async(contents, **kwargs)
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                logger.error(f"Error generating text with Google Gemini model {model_name}: {e}")
                return None
        else:
            logger.warning(f"Unsupported LLM model: {model_name}")
            return None

    async def summarize_text(self, text: str, summarization_prompt: str, **kwargs) -> str | None:
        """
        Generates a summary of the given text using the Gemini Pro model.
        :param text: The text to be summarized.
        :param summarization_prompt: The prompt to guide the summarization.
        :param kwargs: Additional parameters for the LLM API call.
        :return: Generated summary or None if an error occurs.
        """
        full_prompt = f"{summarization_prompt}\n\nText to summarize:\n{text}"
        return await self.generate_text(model_name="gemini-2.5-flash", prompt=full_prompt, **kwargs)

# Initialize the LLMServiceManager globally
llm_service_manager = LLMServiceManager()
