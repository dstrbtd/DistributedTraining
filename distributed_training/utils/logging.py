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


# ------------------------------
# Loki Handler with Error Safety
# ------------------------------
class LokiHandler(logging_loki.LokiHandler):
    def handleError(self, record):
        logging.getLogger(__name__).error("Loki logging error", exc_info=True)
        # No emitter.close() here — keeps retry alive


# ------------------------------
# JSON Formatter for Loki
# ------------------------------
class JSONFormatter(logging.Formatter):
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
        # try:
        #     msg = "".join(record.getMessage().split(" - ")[1:])
        # except Exception:
        msg = record.getMessage()

        # # Terminal-style formatting (without colors)
        # pretty_msg = f"[{record.levelname}] {msg}"
        # if record.name.startswith("bittensor"):
        #     pretty_msg = bt_format.format(record)  # Keep emoji mappings

        log_record = {
            "level": record.levelname.lower(),
            # "pretty": pretty_msg,
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


# ------------------------------
# Hivemind Filter
# ------------------------------
def hive_log_filter(record):
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
    Sets up:
    - Bittensor terminal logging
    - Loki via queue
    - Hivemind logs to file
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
        auth=("944477", os.environ["LOKI_KEY"]),
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
