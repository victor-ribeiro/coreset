import os
from pathlib import Path

ROUTE = {
    "DATA_HOME": Path(os.environ.get("DATA_HOME")),
    "EXPERIMENTS_HOME": Path(os.environ.get("EXPERIMENTS_HOME")),
}

if __name__ == "__main__":
    pass
