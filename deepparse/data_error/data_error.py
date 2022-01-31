class DataError(Exception):
    """
    User error when data is not construct as expected.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
