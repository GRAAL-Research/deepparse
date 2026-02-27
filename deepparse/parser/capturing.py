import sys
from io import StringIO
from types import TracebackType
from typing import Optional, Type


class Capturing(list):
    """
    Decorator function to capture the output of a function.
    """

    def __enter__(self) -> "Capturing":
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout
