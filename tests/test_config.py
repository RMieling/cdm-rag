from api.config import get_config


def test_default_config_load():
    get_config.cache_clear()

    config = get_config()
    assert config.prompts.generate_system_prompt is not None
    assert config.NEO4J_URI == "bolt://neo4j:7687"
    assert config.TEMPERATURE == 0.1


def test_app_config_env_overrides(monkeypatch):
    """
    Test that environment variables override the defaults.
    Use monkeypatch to simulate setting environment variables in a terminal.
    """
    get_config.cache_clear()

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-123")
    monkeypatch.setenv("ENVIRONMENT", "TEST")

    # Should pick up the monkeypatched variables
    config = get_config()

    print(config)

    assert config.LLM_PROVIDER == "openai"
    assert config.OPENAI_API_KEY == "sk-fake-test-key-123"
    assert config.ENVIRONMENT == "TEST"
