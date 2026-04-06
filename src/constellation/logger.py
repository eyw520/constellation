import logging
import logging.config
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

_logging_conf = Path("logging.conf")
if _logging_conf.exists():
    logging.config.fileConfig(_logging_conf)
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

LOGGER = logging.getLogger()
