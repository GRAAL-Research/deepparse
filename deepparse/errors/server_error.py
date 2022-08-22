class ServerError(Exception):
    """
    User error when Deepparse server is not responding.
    """

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)  # pragma: no cover
