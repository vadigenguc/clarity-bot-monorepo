import json
import os
import logging

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

def load_prompt(prompt_name: str) -> str:
    """
    Loads a prompt from a JSON file in the backend/prompts directory.
    :param prompt_name: The name of the prompt file (e.g., "relevance_prompt").
    :return: The prompt string.
    :raises FileNotFoundError: If the prompt file does not exist.
    :raises KeyError: If the 'prompt' key is missing in the JSON file.
    """
    file_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
            if 'prompt' not in prompt_data:
                raise KeyError(f"'{prompt_name}.json' is missing the 'prompt' key.")
            return prompt_data['prompt']
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {file_path}")
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from prompt file {file_path}: {e}")
        raise ValueError(f"Invalid JSON in prompt file: {file_path}")
    except KeyError as e:
        logger.error(f"Key error in prompt file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading prompt {prompt_name} from {file_path}: {e}")
        raise
