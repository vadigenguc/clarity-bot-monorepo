import os
import json
import logging
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

def get_gcp_credentials():
    """Constructs GCP credentials from environment variables."""
    gcp_credentials_json = os.environ.get("GCP_CREDENTIALS_JSON")
    if gcp_credentials_json:
        logger.info("Found GCP_CREDENTIALS_JSON environment variable.")
        try:
            credentials_info = json.loads(gcp_credentials_json)
            logger.info(f"Successfully parsed GCP credentials for service account: {credentials_info.get('client_email')}")
            return service_account.Credentials.from_service_account_info(credentials_info)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse GCP_CREDENTIALS_JSON: {e}", exc_info=True)
            return None
    logger.warning("GCP_CREDENTIALS_JSON not found. Falling back to default credentials.")
    return None # Fallback to default credential discovery
