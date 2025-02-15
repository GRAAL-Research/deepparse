class DataError(Exception):
    """
    User error occurs when the data structure is not as expected.
    """

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)  # pragma: no cover
