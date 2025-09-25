import pytest
import os
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session", autouse=True)
def set_env_vars(tmp_path_factory):
    """Set environment variables for tests."""
    os.environ["OPENAI_API_KEY"] = "test_openai_api_key"
    os.environ["GCP_PROJECT_ID"] = "test-gcp-project"
    os.environ["GCP_LOCATION"] = "europe-west1"
    os.environ["SUPABASE_URL"] = "http://test-supabase.com"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_supabase_key"
    os.environ["SLACK_BOT_TOKEN"] = "xoxb-test"
    os.environ["SLACK_SIGNING_SECRET"] = "test_secret"

    # Create a dummy prompts directory for testing
    prompts_dir = tmp_path_factory.mktemp("prompts")
    os.environ["PROMPTS_DIR"] = str(prompts_dir)

    # Create dummy prompt files
    with open(prompts_dir / "relevance_prompt.json", "w") as f:
        f.write('{"prompt": "relevance_prompt_text"}')
    with open(prompts_dir / "summarization_prompt.json", "w") as f:
        f.write('{"prompt": "summarization_prompt_text"}')
    with open(prompts_dir / "qna_prompt.json", "w") as f:
        f.write('{"prompt": "qna_prompt_text"}')
    with open(prompts_dir / "jira_ticket_prompt.json", "w") as f:
        f.write('{"prompt": "jira_ticket_prompt_text"}')

    # Mock GCP client initializations to prevent actual GCP calls during tests
    with patch('backend.services.llm_service.aiplatform.init', MagicMock()), \
         patch('google.cloud.pubsub_v1.PublisherClient', MagicMock()), \
         patch('google.cloud.pubsub_v1.SubscriberClient', MagicMock()), \
         patch('google.auth.default') as mock_google_auth_default:
        mock_google_auth_default.return_value = (MagicMock(), "test-project")
        yield
