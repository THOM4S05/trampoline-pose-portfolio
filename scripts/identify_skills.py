"""Run the separate identification baseline directly from a checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trampoline_pose.skill_identification import main

if __name__ == "__main__":
    raise SystemExit(main())
