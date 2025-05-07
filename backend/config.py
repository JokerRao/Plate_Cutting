from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """
    配置类，优先级从高到低：
    1. 环境变量（.env 文件中的配置）
    2. 代码中的默认值（这里的默认值）
    """
    # Server Configuration
    HOST: str = "127.0.0.1"  # 默认值，可通过环境变量 HOST 覆盖
    PORT: int = 8000  # 默认值，可通过环境变量 PORT 覆盖
    RELOAD: bool = True  # 默认值，可通过环境变量 RELOAD 覆盖
    LOG_LEVEL: str = "debug"  # 默认值，可通过环境变量 LOG_LEVEL 覆盖
    DEBUG: bool = True  # 默认值，可通过环境变量 DEBUG 覆盖
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000"
    ]  # 默认值，可通过环境变量 CORS_ORIGINS 覆盖
    CORS_ALLOW_CREDENTIALS: bool = True  # 默认值，可通过环境变量 CORS_ALLOW_CREDENTIALS 覆盖
    CORS_ALLOW_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]  # 默认值，可通过环境变量 CORS_ALLOW_METHODS 覆盖
    CORS_ALLOW_HEADERS: list[str] = ["*"]  # 默认值，可通过环境变量 CORS_ALLOW_HEADERS 覆盖
    CORS_EXPOSE_HEADERS: list[str] = ["*"]  # 默认值，可通过环境变量 CORS_EXPOSE_HEADERS 覆盖
    CORS_MAX_AGE: int = 3600  # 默认值，可通过环境变量 CORS_MAX_AGE 覆盖
    
    # Concurrency Configuration
    WORKERS: int = 1  # 默认值，可通过环境变量 WORKERS 覆盖
    BACKLOG: int = 2048  # 默认值，可通过环境变量 BACKLOG 覆盖
    LIMIT_CONCURRENCY: int = 100  # 默认值，可通过环境变量 LIMIT_CONCURRENCY 覆盖
    LIMIT_RATE: Optional[str] = "5/second"  # 默认值，可通过环境变量 LIMIT_RATE 覆盖
    TIMEOUT: int = 300  # 默认值，可通过环境变量 TIMEOUT 覆盖

    # Supabase Configuration
    SUPABASE_URL: Optional[str] = None  # 默认值，可通过环境变量 SUPABASE_URL 覆盖
    SUPABASE_KEY: Optional[str] = None  # 默认值，可通过环境变量 SUPABASE_KEY 覆盖

    # API Configuration
    API_VERSION: str = "1.0.0"  # 默认值，可通过环境变量 API_VERSION 覆盖
    API_TITLE: str = "Plate Cutting API"  # 默认值，可通过环境变量 API_TITLE 覆盖
    API_DESCRIPTION: str = "API for optimizing plate cutting patterns"  # 默认值，可通过环境变量 API_DESCRIPTION 覆盖

    # Logging Configuration
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # 默认值，可通过环境变量 LOG_FORMAT 覆盖

    # Default Parameters
    DEFAULT_SAW_BLADE: int = 4  # 默认值，可通过环境变量 DEFAULT_SAW_BLADE 覆盖
    DEFAULT_TOLERANCE: int = 30  # 默认值，可通过环境变量 DEFAULT_TOLERANCE 覆盖
    DEFAULT_ATTACHMENT: int = 100  # 默认值，可通过环境变量 DEFAULT_ATTACHMENT 覆盖
    DEFAULT_RATIO: float = 0.4  # 默认值，可通过环境变量 DEFAULT_RATIO 覆盖
    DEFAULT_SHOW_AREA: int = 120000  # 默认值，可通过环境变量 DEFAULT_SHOW_AREA 覆盖

    class Config:
        env_file = ".env.local"  # 指定环境变量文件
        case_sensitive = True  # 环境变量名称区分大小写

@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例，使用缓存避免重复加载
    """
    return Settings() 