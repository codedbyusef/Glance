import sys
from pathlib import Path

# Tests import the package and the rig directly: run pytest from agent/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
