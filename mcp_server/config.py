from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_base_url: str = "http://127.0.0.1:8000"
    http_timeout_sec: float = 30.0
    search_limit_max: int = 200
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="MULTIMEDIA_")


settings = Settings()
