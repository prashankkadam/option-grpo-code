"""tinyframe: a miniature, fully typed dataframe library.

It exists so that the whole Option-GRPO pipeline (graph construction, option
discovery, task and property-test generation, sandboxed execution, training and
evaluation) can be exercised end to end on a laptop without installing a real SDK.
"""

from tinyframe._io import from_records, read_csv, read_json
from tinyframe.expr import Expr, col, lit
from tinyframe.frame import Frame, GroupedFrame
from tinyframe.store import Database, Query, StoredTable, connect
from tinyframe.table import Table

__all__ = [
    "Database",
    "Expr",
    "Frame",
    "GroupedFrame",
    "Query",
    "StoredTable",
    "Table",
    "col",
    "connect",
    "from_records",
    "lit",
    "read_csv",
    "read_json",
]

__version__ = "0.1.0"
