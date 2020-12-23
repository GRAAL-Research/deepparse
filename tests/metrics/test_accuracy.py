# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable

from unittest import TestCase

import torch


class AccuracyTest(TestCase):

    def setUp(self) -> None:
        self.a_device = "cpu"
        self.ground_truth = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_device)

    def test_whenAllPredictionAreOk_thenAccuracyIs100(self):
        pass
