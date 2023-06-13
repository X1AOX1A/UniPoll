import os
import sys
from loguru import logger

def get_logger(log_dir, log_name):
    FMT = """[<level>{level}</level>|{name}:{line}]  \
<green>{time:YYYY-MM-DD HH:mm:ss}</green> >> \
<level>{message}</level>"""

    log_file = os.path.join(log_dir, log_name)
    # overwrite log file
    with open(log_file, "w") as f:
        pass
    # remove original stream handler
    logger.remove()
    # add custom stream handler
    logger.add(sys.stderr, level="INFO", format=FMT)
    # add custom file handler
    logger.add(log_file, level="DEBUG", format=FMT, enqueue=True)
    return logger