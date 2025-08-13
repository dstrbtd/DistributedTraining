import os
import json
import shutil
import logging
import bittensor as bt
import logging_loki
import traceback

from dotenv import load_dotenv
from hivemind.utils.logging import use_hivemind_log_handler
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue

from distributed_training import __version__, __spec_version__
from bittensor.utils.btlogging import format as bt_format

load_dotenv()


class LokiHandler(logging_loki.LokiHandler):
    """
    Custom Loki logging handler that safely handles errors.

    Overrides `handleError` to log any exceptions that occur during logging
    to Loki, without shutting down the emitter. This ensures that retry logic
    remains active instead of terminating on the first failure.
    """

    def handleError(self, record):
        logging.getLogger(__name__).error("Loki logging error", exc_info=True)
        # No emitter.close() here — keeps retry alive


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for Loki ingestion.

    Adds extra metadata about the miner, such as:
    - network and netuid
    - hotkey
    - neuron type
    - IP/port
    - Bittensor and spec version numbers
    - UID

    If exception info is present, it is included as a formatted string.
    """

    def __init__(self, miner):
        self.network = miner.config.subtensor.network
        self.netuid = miner.config.netuid
        self.hotkey = miner.wallet.hotkey.ss58_address
        self.version = __version__
        self.spec_version = __spec_version__
        self.run_id = None
        self.ip = (
            miner.config.axon.ip
            if miner.config.axon.ip != "[::]"
            else bt.utils.networking.get_external_ip()
        )
        self.port = miner.config.axon.port
        self.uid = miner.uid
        self.neuron_type = "validator"

    def format(self, record):
        msg = record.getMessage()

        log_record = {
            "level": record.levelname.lower(),
            "module": record.module,
            "func_name": record.funcName,
            "thread": record.threadName,
            "netuid": self.netuid,
            "network": self.network,
            "neuron_type": self.neuron_type,
            "hotkey": self.hotkey,
            "uid": self.uid,
            "ip": self.ip,
            "port": self.port,
            "message": msg,
            "filename": record.filename,
            "lineno": record.lineno,
            "version": self.version,
            "spec_version": self.spec_version,
        }

        if record.exc_info:
            log_record["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_record)


def hive_log_filter(record):
    """
    Filters out noisy Hivemind loggers that are not relevant to application output.

    Returns:
        bool: True if the record should be logged, False otherwise.
    """
    return record.name not in {
        "hivemind.dht.protocol",
        "hivemind.optim.progress_tracker",
        "hivemind.p2p.p2p_daemon_bindings.control",
    }


# ------------------------------
# Main Setup Function
# ------------------------------
def setup_logging(self, local_logfile="logs_mylogfile.txt", config=None):
    """
    Configure and start logging for the distributed training miner.

    This includes:
    - Bittensor terminal logging with custom emoji mapping
    - Loki logging via a background queue listener
    - File logging for Hivemind output (filtered)
    - Disabling noisy loggers and default Hivemind handlers

    Args:
        self: Miner instance containing config, wallet, and UID.
        local_logfile (str): Path to the local log file.
        config: Optional Bittensor config object for logging.
    """

    # Configure Bittensor terminal output
    bt_format.emoji_map.update(
        {
            ":rocket:": "🚀",
            ":lock:": "🔒",
            ":unlock:": "🔓",
            ":lightning:": "⚡",
            ":error:": "❗",
            ":info:": "ℹ️",
            ":idle:": "😴",
            ":network:": "🌐",
            ":memory:": "💾",
            ":training:": "🏋️",
            ":progress:": "📈",
            ":wait:": "⏳",
            ":clock:": "⏱️",
            ":signal:": "📶",
            ":upload:": "🔼",
            ":broadcast:": "📡",
            ":sync:": "🔄",
            ":send:": "📤",
            ":receive:": "📥",
            ":pages:": "📑",
        }
    )
    bt.logging(config=config or bt.config())

    bt_logger = logging.getLogger("bittensor")

    # Default to INFO if no flags are set
    if not (
        getattr(config.logging, "debug", False)
        or getattr(config.logging, "trace", False)
        or getattr(config.logging, "info", False)
    ):
        bt_logger.setLevel(logging.INFO)

    # Prepare root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Loki handler with extra labels
    loki_handler = LokiHandler(
        url="https://logs-prod-006.grafana.net/loki/api/v1/push",
        tags={
            "application": "distributed_training",
            "level": "dynamic",  # Will be overridden dynamically
            "hotkey": self.wallet.hotkey.ss58_address,
            "netuid": str(self.config.netuid),
        },
        auth=("944477", os.getenv("LOKI_KEY")),
        version="1",
    )
    loki_handler.setLevel(logging.DEBUG)
    loki_handler.setFormatter(JSONFormatter(self))

    # Wrap emit so level label matches log level
    original_emit = loki_handler.emit

    def dynamic_label_emit(record):
        loki_handler.emitter.tags["level"] = record.levelname.lower()
        original_emit(record)

    loki_handler.emit = dynamic_label_emit

    # File handler for Hivemind
    if os.path.exists(local_logfile):
        shutil.copyfile(local_logfile, local_logfile.replace(".txt", "_archive.txt"))
        os.remove(local_logfile)

    file_handler = logging.FileHandler(local_logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(hive_log_filter)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Setup queue logging
    log_queue = Queue(-1)
    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, loki_handler, file_handler)
    listener.start()

    # Disable noisy hivemind default logging
    use_hivemind_log_handler("nowhere")

    # Disable propagation for other loggers
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if name not in ["bittensor"]:
                logger.propagate = False
