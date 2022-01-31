class DataError(Exception):
    """
    User error when data is not construct as expected.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __str__(self):
        return repr(self.value)  # pragma: no cover
