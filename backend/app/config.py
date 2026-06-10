import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"
    backend_cors_origins: str = "http://localhost:5174,http://127.0.0.1:5174"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_trust_env: bool = True
    openai_timeout_seconds: float = 180.0
    openai_max_retries: int = 0
    openai_max_concurrency: int = 5
    openai_fallback_api_key: str | None = None
    openai_fallback_base_url: str | None = None
    openai_fallback_model: str | None = None
    openai_main_max_concurrency: int | None = None
    openai_qa_max_concurrency: int | None = None
    openai_retry_max_concurrency: int = 1

    tavily_api_key: str | None = None
    tavily_max_results: int = 5
    duckduckgo_max_results: int = 6
    search_browser_disable_proxy: bool = True
    search_query_concurrency: int = 5
    max_queries_per_generation: int = 5
    field_search_iterations_per_competitor_tool: int = 3
    card_search_iterations_per_object: int = 3
    card_object_concurrency: int = 3
    card_patch_concurrency: int | None = None
    candidate_qa_concurrency: int = 5
    tool_execution_concurrency: int = 3

    storage_backend: Literal["sqlite", "postgres", "json"] = "sqlite"
    sqlite_path: Path = BACKEND_DIR / "data" / "app.db"
    json_storage_dir: Path = BACKEND_DIR / "data" / "json"
    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/competitor_agent"

    max_revision_loops: int = 2

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        value = self.backend_cors_origins.strip()
        if not value:
            return []
        if value.startswith("["):
            parsed = json.loads(value)
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def data_dir(self) -> Path:
        if self.storage_backend == "json":
            return self.json_storage_dir.parent
        if self.storage_backend == "sqlite":
            return self.sqlite_path.parent
        return BACKEND_DIR / "data"

    @property
    def task_output_dir(self) -> Path:
        return self.data_dir / "task_runs"

    @property
    def effective_openai_main_max_concurrency(self) -> int:
        if self.openai_main_max_concurrency is not None:
            return max(1, int(self.openai_main_max_concurrency))
        return min(3, max(1, int(self.openai_max_concurrency)))

    @property
    def effective_openai_qa_max_concurrency(self) -> int:
        if self.openai_qa_max_concurrency is not None:
            return max(1, int(self.openai_qa_max_concurrency))
        return 1

    @property
    def effective_openai_retry_max_concurrency(self) -> int:
        return max(1, int(self.openai_retry_max_concurrency))

    @property
    def effective_card_patch_concurrency(self) -> int:
        if self.card_patch_concurrency is not None:
            return max(1, int(self.card_patch_concurrency))
        return self.effective_openai_main_max_concurrency


@lru_cache
def get_settings() -> Settings:
    return Settings()
