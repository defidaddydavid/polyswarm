"""
Data source plugins — each file registers its sources on import.
Import this package to auto-register all built-in sources.
"""

from data.sources.market import *
from data.sources.derivatives_src import *
from data.sources.onchain_src import *
from data.sources.sentiment import *
from data.sources.prediction import *
