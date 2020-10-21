from pymagnitude import Magnitude


class MagnitudeWrapper(Magnitude):
    """
    Wrapper to standardize the querying of a word using the [].
    """

    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, item):
        """
        Overload the getitem method since magnitude model use query instead of [] like fastText "normal" model.
        """
        return self.query(item)
