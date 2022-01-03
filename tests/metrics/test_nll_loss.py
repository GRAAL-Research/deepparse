# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import unittest
from unittest import TestCase

import torch

from deepparse.metrics import nll_loss


class NLLLossTest(TestCase):
    def setUp(self) -> None:
        self.a_device = torch.device("cpu")

        # 2 address of six elements each
        self.ground_truth = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_device)
        self.a_prediction_tensor = torch.tensor(
            [
                [
                    [
                        -2.2366e-03,
                        -7.1717e00,
                        -1.2499e01,
                        -6.7893e00,
                        -1.3376e01,
                        -8.6677e00,
                        -9.3836e00,
                        -9.4463e00,
                        -1.9178e01,
                    ],
                    [
                        -1.7175e01,
                        -7.4981e-02,
                        -1.5088e01,
                        -7.0761e00,
                        -2.7295e00,
                        -5.0936e00,
                        -1.2763e01,
                        -1.2705e01,
                        -4.2618e01,
                    ],
                ],
                [
                    [
                        -1.3740e01,
                        -3.1232e-05,
                        -2.0795e01,
                        -1.0409e01,
                        -2.6400e01,
                        -2.3393e01,
                        -2.1003e01,
                        -2.1134e01,
                        -3.9064e01,
                    ],
                    [
                        -3.3409e00,
                        -2.4993e00,
                        -1.7771e01,
                        -1.2612e-01,
                        -1.5750e01,
                        -6.9642e00,
                        -1.5144e01,
                        -1.5204e01,
                        -3.5160e01,
                    ],
                ],
                [
                    [
                        -1.6033e01,
                        -3.2526e00,
                        -1.6328e01,
                        -3.9443e-02,
                        -1.6613e01,
                        -1.7416e01,
                        -1.8283e01,
                        -1.8346e01,
                        -3.1463e01,
                    ],
                    [
                        -9.6402e00,
                        -8.4887e00,
                        -1.7941e01,
                        -5.0700e00,
                        -6.8180e-03,
                        -8.3299e00,
                        -1.6193e01,
                        -1.6284e01,
                        -3.3004e01,
                    ],
                ],
                [
                    [
                        -1.3168e01,
                        -1.4217e01,
                        -1.3901e01,
                        -1.2135e01,
                        -9.6559e-06,
                        -1.4170e01,
                        -1.9320e01,
                        -1.9513e01,
                        -3.6137e01,
                    ],
                    [
                        -1.3708e01,
                        -1.5111e01,
                        -1.7616e01,
                        -1.7709e01,
                        -1.5568e-04,
                        -8.7770e00,
                        -1.7401e01,
                        -1.7421e01,
                        -1.9826e01,
                    ],
                ],
                [
                    [
                        -1.2694e01,
                        -1.7848e01,
                        -1.5564e01,
                        -2.3230e01,
                        -1.0867e01,
                        -2.7418e-05,
                        -1.7265e01,
                        -1.7370e01,
                        -1.2187e01,
                    ],
                    [
                        -1.8689e01,
                        -1.8121e01,
                        -2.0229e01,
                        -2.5230e01,
                        -6.5044e00,
                        -1.6420e01,
                        -1.9653e01,
                        -1.9648e01,
                        -1.4981e-03,
                    ],
                ],
                [
                    [
                        -1.7753e01,
                        -1.9364e01,
                        -1.9066e01,
                        -2.8301e01,
                        -1.8600e01,
                        -1.0906e01,
                        -2.0326e01,
                        -2.0414e01,
                        -1.8358e-05,
                    ],
                    [
                        -2.4182e01,
                        -2.5269e01,
                        -2.6161e01,
                        -3.2104e01,
                        -1.4497e01,
                        -2.2898e01,
                        -2.4870e01,
                        -2.4851e01,
                        -4.7684e-07,
                    ],
                ],
            ],
            device=self.a_device,
            requires_grad=True,
        )
        self.a_loss = torch.tensor(37.2189, device=self.a_device)

        # 2 address of two element each
        self.a_short_ground_truth = torch.tensor([[0, 1], [1, 0]], device=self.a_device)

    def test_givenAPredictionTensor_whenNLLLossPerTag_thenLossIsOk(self):
        # need to convert to list and use float since not working almost equal for tensor
        actual = nll_loss(self.a_prediction_tensor, self.ground_truth).detach().tolist()
        expected = self.a_loss.tolist()
        self.assertAlmostEqual(expected, actual, delta=5)

    def test_givenAPerfectPredictionTensor_whenNLLLossPerTag_thenLossIs0(self):
        # Tags prediction value are "inverted" to mimic the log
        first_token_first_element_of_the_batch = [
            0.0,
            1.0,
        ]  # the predicted token is the first class
        first_token_second_element_of_the_batch = [
            1.0,
            0.0,
        ]  # the predicted token is the second class
        second_token_first_element_of_the_batch = [
            1.0,
            0.0,
        ]  # the predicted token is the second class
        second_token_second_element_of_the_batch = [
            0.0,
            1.0,
        ]  # the predicted token is the first class
        predict_tensor = torch.tensor(
            [
                [
                    first_token_first_element_of_the_batch,
                    first_token_second_element_of_the_batch,
                ],
                [
                    second_token_first_element_of_the_batch,
                    second_token_second_element_of_the_batch,
                ],
            ],
            device=self.a_device,
        )

        actual = nll_loss(predict_tensor, self.a_short_ground_truth).detach().tolist()
        expected = torch.tensor(1, device=self.a_device).tolist()
        self.assertAlmostEqual(expected, actual, delta=5)

    def test_givenACompletelyWrongPredictionTensor_whenNLLLossPerTag_thenLossIsMinus2(
        self,
    ):
        # Tags prediction value are "inverted" to mimic the log
        first_token_first_element_of_the_batch = [
            1.0,
            0.0,
        ]  # the predicted token is the second class
        first_token_second_element_of_the_batch = [
            0.0,
            1.0,
        ]  # the predicted token is the first class
        second_token_first_element_of_the_batch = [
            0.0,
            1.0,
        ]  # the predicted token is the first class
        second_token_second_element_of_the_batch = [
            1.0,
            0.0,
        ]  # the predicted token is the second class
        predict_tensor = torch.tensor(
            [
                [
                    first_token_first_element_of_the_batch,
                    first_token_second_element_of_the_batch,
                ],
                [
                    second_token_first_element_of_the_batch,
                    second_token_second_element_of_the_batch,
                ],
            ],
            device=self.a_device,
        )

        actual = nll_loss(predict_tensor, self.a_short_ground_truth).detach().tolist()
        expected = torch.tensor(0.0, device=self.a_device)
        self.assertAlmostEqual(expected, actual, delta=5)


if __name__ == "__main__":
    unittest.main()
