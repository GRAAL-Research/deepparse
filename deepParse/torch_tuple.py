class TorchTuple(tuple):
    """
    Wrapper class to add to() PyTorch method to a tuple. This allow us to load to a device a tuple of tensor and tuple
    elements such at the one generated from ~converter.data_padding.bpemb_data_padding.
    """

    def to(self, device):
        pass
