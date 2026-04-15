import os

import pytest

from api.utils.utils import check_ollama_models_ready

# Check if test is running in GitHub Actions
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# TODO still requires running ollama server
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipping local Ollama integration test in CI.")
def test_local_ollama_models_ready():
    """
    This test checks if the required Ollama models are downloaded and ready on a local development machine.
    """
    endpoint = "http://localhost:11434"
    required_models = ["mistral", "nomic-embed-text"]

    is_ready = check_ollama_models_ready(endpoint, required_models)
    assert is_ready is True, f"Required models {required_models} are missing from local Ollama."

    is_ready = check_ollama_models_ready(endpoint, ["llama3-70b"])
    assert is_ready is False, "Model 'llama3-70b' should not be found in local Ollama, but it was."
