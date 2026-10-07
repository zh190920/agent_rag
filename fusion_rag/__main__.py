"""允许 ``python -m fusion_rag <command>`` 直接调用 CLI。"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
