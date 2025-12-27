import logging
import os
import time
import shutil

from logging.handlers import RotatingFileHandler

RAY_LOG_CLEANER_LOG_PATH_ENV="RAY_LOG_CLEANER_LOG_PATH"
RAY_LOG_DIR_ENV="RAY_LOG_DIR"
RAY_LOG_TO_KEEP_IN_DAY_ENV="RAY_LOG_TO_KEEP_IN_DAY"
logger = None


def setup_logger(max_bytes=100 * 1024 * 1024, backup_count=3):
    global logger
    if logger is not None:
        return logger

    logger = logging.getLogger("ray_logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.getenv(RAY_LOG_CLEANER_LOG_PATH_ENV)
        handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        formatter = logging.Formatter('%(asctime)s %(levelname)-6s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def is_over_retention(timestamp):
    days = int(os.getenv(RAY_LOG_TO_KEEP_IN_DAY_ENV))
    retention = days * 24 * 3600
    return (time.time() - timestamp) > retention


def clean_directory(log):
    log.info("Start to clean up ray log")
    try:
        ray_log_dir = os.getenv(RAY_LOG_DIR_ENV)

        if not os.path.isdir(ray_log_dir):
            log.warn(f"Log dir does NOT exist: {ray_log_dir}")
            return

        session_latest = os.path.join(ray_log_dir, "session_latest")
        protected_path = None

        if os.path.islink(session_latest):
            target = os.readlink(session_latest)
            dirname = os.path.basename(target)
            protected_path = os.path.join(ray_log_dir, dirname)

        for entry in os.scandir(ray_log_dir):
            if entry.name == "session_latest":
                continue

            dir_realpath = os.path.realpath(entry.path)
            if protected_path and dir_realpath == protected_path:
                continue

            dir_stat = entry.stat()
            if is_over_retention(dir_stat.st_mtime):
                shutil.rmtree(entry.path)

    except Exception as e:
        log.error(f"clean up ray log failed, exception: {e}")
    log.info("Finish to clean up ray log")


if __name__ == "__main__":
    log = setup_logger()
    clean_directory(log)
