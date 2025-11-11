import logging
import os

def get_logger(name: str):
    """Create and return a configured logger."""
    os.makedirs("logs", exist_ok=True)

    log_file = os.path.join("logs", "app.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Also print logs to console for visibility
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging.getLogger(name)