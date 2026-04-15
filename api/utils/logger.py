import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional


def setup_logging(
    log_path_str: Optional[str] = None,
    log_level: Optional[str] = None,
    enable_file_logging: bool = True,
) -> Dict[str, logging.Logger]:
    """
    Configure logging for the application with persistent storage.

    Args:
        log_dir: Directory for log files (default: logs/ or LOG_DIR env var)
        log_level: Logging level (default: INFO or LOG_LEVEL env var)
        enable_file_logging: Whether to enable file logging (default: True)

    Returns:
        Dict[str, logging.Logger]: Dictionary of configured loggers

    """

    # Load configuration from environment variables if not provided
    log_path_str = os.getenv("LOG_DIR", "logs") if log_path_str is None else log_path_str
    log_level = os.getenv("LOG_LEVEL", "INFO") if log_level is None else log_level

    log_path = Path(log_path_str)

    # Create logs directory if it doesn't exist (important for Docker volumes)
    try:
        log_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        warnings.warn(
            f"Warning: Cannot create log directory {log_path}. Falling back to console logging only.", stacklevel=2
        )
        enable_file_logging = False

    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure handlers
    handlers: list[logging.Handler] = []

    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    handlers.append(console_handler)

    # File handlers (if enabled and possible)
    if enable_file_logging:
        try:
            # Main application log
            file_handler = logging.FileHandler(log_path / "cdm_rag.log", encoding="utf-8")
            file_handler.setFormatter(detailed_formatter)
            handlers.append(file_handler)

            # Error log (only errors and critical)
            error_handler = logging.FileHandler(log_path / "cdm_rag_errors.log", encoding="utf-8")
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            handlers.append(error_handler)

        except Exception as e:
            print(f"Warning: File logging failed ({e}). Using console logging only.")

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Configure specific loggers for our modules
    logger_names = ["main", "rag", "ingestion", "parse_cdm", "api", "frontend", "docker"]

    loggers: Dict[str, logging.Logger] = {}
    for name in logger_names:
        logger = logging.getLogger(f"cdm_rag.{name}")
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        loggers[name] = logger

    # Log startup information
    main_logger = loggers["main"]
    main_logger.info("CDM RAG logging initialized")
    main_logger.info(f"Log level: {log_level}")
    main_logger.info(f"Log directory: {log_path.absolute()}")
    main_logger.info(f"File logging: {'enabled' if enable_file_logging else 'disabled'}")

    # Docker-specific logging
    if os.getenv("DOCKER_CONTAINER"):
        docker_logger = loggers["docker"]
        docker_logger.info("Running in Docker container")
        docker_logger.info(f"Log volume mounted: {log_path.absolute()}")

    return loggers


# Initialize loggers with environment-aware configuration
LOGGERS: Dict[str, logging.Logger] = setup_logging()

# Export loggers for easy import across the app
main_logger: logging.Logger = LOGGERS["main"]
rag_logger: logging.Logger = LOGGERS["rag"]
parse_cdm_logger: logging.Logger = LOGGERS["parse_cdm"]
ingestion_logger: logging.Logger = LOGGERS["ingestion"]
api_logger: logging.Logger = LOGGERS["api"]
frontend_logger: logging.Logger = LOGGERS["frontend"]
docker_logger: logging.Logger = LOGGERS["docker"]
