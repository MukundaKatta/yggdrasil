"""Allow running yggdrasil as a module: python -m yggdrasil."""

import sys

from yggdrasil.cli import main

sys.exit(main())
