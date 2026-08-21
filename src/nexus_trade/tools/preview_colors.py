import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus_trade.utils.colored_logging import ColoredFormatter, enable_ansi

_LINE_RE = re.compile(r"^\d{2,4}-\d{2}-\d{2} [\d:,]+ - (\w+) - (.*)$")


def main(log_path: str) -> None:
    enable_ansi()

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger("preview")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    with Path(log_path).open() as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            match = _LINE_RE.match(line)
            if not match:
                continue
            level_name, message = match.groups()
            level = logging.getLevelName(level_name)
            if not isinstance(level, int):
                level = logging.INFO
            logger.log(level, message)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample.log")
