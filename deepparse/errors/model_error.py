class FastTextModelError(Exception):
    """
    User error when user uses a FastText-like model on an OS that does not support properly multithreading.
    """

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)  # pragma: no cover
