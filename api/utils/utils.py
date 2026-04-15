import requests


def check_ollama_models_ready(endpoint: str, required_models: list) -> bool:
    """
    Pings the Ollama API and checks if all required models are downloaded.
    Raises an exception if the server is completely unreachable.
    """
    response = requests.get(f"{endpoint}/api/tags", timeout=5)

    data = response.json()
    downloaded_models = [m["name"] for m in data.get("models", [])]

    # Returns True only if EVERY required model is found in the downloaded list
    return all(any(req_model in m for m in downloaded_models) for req_model in required_models)
