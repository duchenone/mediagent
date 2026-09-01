import logging
import os
from logging.handlers import TimedRotatingFileHandler

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# 日志保留天数, 超出的历史日志由轮转器自动删除
LOG_BACKUP_DAYS = 14


def _file_log_level() -> int:
    """文件日志级别: 默认 INFO(医疗场景避免问诊内容落盘), 调试时设 MEDIAGENT_LOG_LEVEL=DEBUG"""
    return getattr(logging, os.environ.get('MEDIAGENT_LOG_LEVEL', 'INFO').upper(), logging.INFO)


def get_logger(
        name: str = "mediagent",
        console_level: int = logging.INFO,
        file_level: int | None = None,
        log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # logger 自身放行, 由 handler 各自过滤

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}.log")

    # 按天轮转, 保留 LOG_BACKUP_DAYS 天, 自动清理旧日志
    file_handler = TimedRotatingFileHandler(
        log_file, when='midnight', backupCount=LOG_BACKUP_DAYS, encoding='utf-8',
    )
    file_handler.setLevel(file_level if file_level is not None else _file_log_level())
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
