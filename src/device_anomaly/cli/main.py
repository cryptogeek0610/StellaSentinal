import logging

from device_anomaly.config.logging_config import setup_logging
from device_anomaly import __version__


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Device Anomaly Service starting up...")
    logger.info("Version: %s", __version__)
    logger.info("No experiments defined yet. This is just a skeleton.")


if __name__ == "__main__":
    main()
