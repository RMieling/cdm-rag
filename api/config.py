from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# load_dotenv()


class PromptsConfig(BaseModel):
    generate_system_prompt: str
    contextualize_system_prompt: str
    cypher_system_prompt: str


class AppConfig(BaseSettings):
    # Neo4j Settings
    NEO4J_URI: str
    NEO4J_USERNAME: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None

    # Provider
    LLM_PROVIDER: str
    # Model temperature
    TEMPERATURE: float
    RAG_K: int

    # Ollama Settings
    OLLAMA_ENDPOINT: str
    OLLAMA_EMBED_MODEL: str
    OLLAMA_LLM_MODEL: str

    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str
    OPENAI_EMBED_MODEL: str

    # Data directory (for local file uploads and vector store persistence)
    DATA_DIR: str

    # Environment & Logging
    ENVIRONMENT: str
    LOG_DIR: str
    LOG_LEVEL: str
    ENABLE_FILE_LOGGING: bool

    prompts: PromptsConfig

    # Read from your .env file
    model_config = SettingsConfigDict(env_file=".env", yaml_file="config/base.yaml", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """
        Define the priority of configuration sources from Highest to Lowest:
        """
        return (
            init_settings,  # 1. Kwargs passed directly in code: AppConfig(LOG_LEVEL="DEBUG")
            env_settings,  # 2. OS Environment variables
            dotenv_settings,  # 3. Variables loaded from the .env file
            YamlConfigSettingsSource(settings_cls),  # 4. Variables loaded from base.yaml
        )


@lru_cache()
def get_config() -> AppConfig:
    """
    Get the application configuration with caching via @lru_cache to ensure it's only loaded once.
    """
    # load prompts separately to avoid issues with nested Pydantic models and YAML parsing
    prompts_path = Path("config/prompts.yaml")
    try:
        with open(prompts_path, "r", encoding="utf-8") as file:
            raw_prompts = yaml.safe_load(file)
            loaded_prompts = PromptsConfig(**raw_prompts)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not find prompts file at {prompts_path}") from e

    config = AppConfig(prompts=loaded_prompts)

    return config


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question about the CDM definitions.")
    session_id: str = Field(
        default="default_session",
        description="Unique ID for the user's chat thread to maintain memory.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The generated answer from the AI.")
    sources: List[str] = Field(
        default_factory=list,
        description="List of document sources used to generate the answer.",
    )
    error: Optional[str] = Field(None, description="Error message, if any.")


if __name__ == "__main__":
    # See the config values when running this file directly
    config = AppConfig()
    print(config.model_dump)
