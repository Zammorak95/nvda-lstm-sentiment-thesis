# src/thesis/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    # Required; will raise a clear error if missing
    stockdata_api_token: SecretStr

    # Look for .env in project root, no prefix, case-insensitive is fine
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )

# Singleton-style settings object
settings = Settings() # type: ignore
