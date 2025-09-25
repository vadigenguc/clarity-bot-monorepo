import pytest
import os
import json
from unittest.mock import patch # Import patch
from backend.utils.prompt_loader import load_prompt

def test_load_prompt_success(tmp_path):
    """Test that a prompt is loaded successfully."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    with open(prompts_dir / "test_prompt.json", "w") as f:
        f.write('{"prompt": "test_prompt_text"}')
    
    with patch('backend.utils.prompt_loader.PROMPTS_DIR', str(prompts_dir)):
        prompt = load_prompt("test_prompt")
        assert prompt == "test_prompt_text"

def test_load_prompt_file_not_found(tmp_path):
    """Test that FileNotFoundError is raised for a missing prompt file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    
    with patch('backend.utils.prompt_loader.PROMPTS_DIR', str(prompts_dir)):
        with pytest.raises(FileNotFoundError):
            load_prompt("non_existent_prompt")

def test_load_prompt_missing_key(tmp_path):
    """Test that KeyError is raised for a missing 'prompt' key."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    with open(prompts_dir / "missing_key_prompt.json", "w") as f:
        f.write('{"wrong_key": "test_prompt_text"}')
    
    with patch('backend.utils.prompt_loader.PROMPTS_DIR', str(prompts_dir)):
        with pytest.raises(KeyError):
            load_prompt("missing_key_prompt")

def test_load_prompt_malformed_json(tmp_path):
    """Test that ValueError is raised for malformed JSON."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    with open(prompts_dir / "malformed_prompt.json", "w") as f:
        f.write('{"prompt": "test_prompt_text"')
    
    with patch('backend.utils.prompt_loader.PROMPTS_DIR', str(prompts_dir)):
        with pytest.raises(ValueError):
            load_prompt("malformed_prompt")
