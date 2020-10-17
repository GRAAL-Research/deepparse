from pymagnitude import Magnitude


class MagnitudeWrapper(Magnitude):
    """
    Wrapper to standardize the querying of a word using the [].
    """

    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, item):
        return self.query(item)
