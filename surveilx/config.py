from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SURVEILX_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/surveilx.db"
    admin_password: str = ""
    secure_cookies: bool = False
    allowed_origin: str = "http://127.0.0.1:8000"
    demo: bool = False
    start_workers: bool = True
    max_cameras: int = 16
    epoch_seconds: float = 1.0
    coverage_seconds: float = 5.0
    confirmation_seconds: float = 2.0
    cooldown_seconds: float = 60.0
    detection_log_seconds: float = 60.0
    retention_days: int = 7
    detector_path: str = ""
    training_python: str = ""
    training_batch_size: int | None = None
    training_yolo_batch_size: int | None = None
    training_event_batch_size: int | None = None
    s3_bucket: str = ""
    s3_endpoint: str = ""


settings = Settings()
