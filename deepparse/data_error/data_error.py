class DataError(Exception):
    """
    User error when data is not construct as expected.
    """

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)  # pragma: no cover
