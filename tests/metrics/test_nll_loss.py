# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import unittest
from unittest import TestCase

import torch

from deepparse.metrics import nll_loss


class NLLLossTest(TestCase):

    def setUp(self) -> None:
        self.a_device = "cpu"

        # 2 address of six elements each
        self.ground_truth = torch.tensor([[0, 1, 1, 4, 5, 8], [1, 0, 3, 8, 0, 0]], device=self.a_device)
        self.a_prediction_tensor = torch.tensor([[[
            -2.2366e-03, -7.1717e+00, -1.2499e+01, -6.7893e+00, -1.3376e+01, -8.6677e+00, -9.3836e+00, -9.4463e+00,
            -1.9178e+01
        ],
                                                  [
                                                      -1.7175e+01, -7.4981e-02, -1.5088e+01, -7.0761e+00, -2.7295e+00,
                                                      -5.0936e+00, -1.2763e+01, -1.2705e+01, -4.2618e+01
                                                  ]],
                                                 [[
                                                     -1.3740e+01, -3.1232e-05, -2.0795e+01, -1.0409e+01, -2.6400e+01,
                                                     -2.3393e+01, -2.1003e+01, -2.1134e+01, -3.9064e+01
                                                 ],
                                                  [
                                                      -3.3409e+00, -2.4993e+00, -1.7771e+01, -1.2612e-01, -1.5750e+01,
                                                      -6.9642e+00, -1.5144e+01, -1.5204e+01, -3.5160e+01
                                                  ]],
                                                 [[
                                                     -1.6033e+01, -3.2526e+00, -1.6328e+01, -3.9443e-02, -1.6613e+01,
                                                     -1.7416e+01, -1.8283e+01, -1.8346e+01, -3.1463e+01
                                                 ],
                                                  [
                                                      -9.6402e+00, -8.4887e+00, -1.7941e+01, -5.0700e+00, -6.8180e-03,
                                                      -8.3299e+00, -1.6193e+01, -1.6284e+01, -3.3004e+01
                                                  ]],
                                                 [[
                                                     -1.3168e+01, -1.4217e+01, -1.3901e+01, -1.2135e+01, -9.6559e-06,
                                                     -1.4170e+01, -1.9320e+01, -1.9513e+01, -3.6137e+01
                                                 ],
                                                  [
                                                      -1.3708e+01, -1.5111e+01, -1.7616e+01, -1.7709e+01, -1.5568e-04,
                                                      -8.7770e+00, -1.7401e+01, -1.7421e+01, -1.9826e+01
                                                  ]],
                                                 [[
                                                     -1.2694e+01, -1.7848e+01, -1.5564e+01, -2.3230e+01, -1.0867e+01,
                                                     -2.7418e-05, -1.7265e+01, -1.7370e+01, -1.2187e+01
                                                 ],
                                                  [
                                                      -1.8689e+01, -1.8121e+01, -2.0229e+01, -2.5230e+01, -6.5044e+00,
                                                      -1.6420e+01, -1.9653e+01, -1.9648e+01, -1.4981e-03
                                                  ]],
                                                 [[
                                                     -1.7753e+01, -1.9364e+01, -1.9066e+01, -2.8301e+01, -1.8600e+01,
                                                     -1.0906e+01, -2.0326e+01, -2.0414e+01, -1.8358e-05
                                                 ],
                                                  [
                                                      -2.4182e+01, -2.5269e+01, -2.6161e+01, -3.2104e+01, -1.4497e+01,
                                                      -2.2898e+01, -2.4870e+01, -2.4851e+01, -4.7684e-07
                                                  ]]],
                                                device=self.a_device,
                                                requires_grad=True)
        self.a_loss = torch.tensor(37.2189, device=self.a_device)

        # 2 address of two element each
        self.a_short_ground_truth = torch.tensor([[0, 1], [1, 0]], device=self.a_device)

    def test_givenAPredictionTensor_whenNLLLossPerTag_thenLossIsOk(self):
        # need to convert to list and use float since not working almost equal for tensor
        actual = nll_loss(self.a_prediction_tensor, self.ground_truth).detach().tolist()
        expected = self.a_loss.tolist()
        self.assertAlmostEqual(expected, actual, delta=5)

    def test_givenAPerfectPredictionTensor_whenNLLLossPerTag_thenLossIs0(self):
        # Tags prediction value are 'inverted' to mimic the log
        first_token_first_element_of_the_batch = [0., 1.]  # the predicted token is the first class
        first_token_second_element_of_the_batch = [1., 0.]  # the predicted token is the second class
        second_token_first_element_of_the_batch = [1., 0.]  # the predicted token is the second class
        second_token_second_element_of_the_batch = [0., 1.]  # the predicted token is the first class
        predict_tensor = torch.tensor(
            [[first_token_first_element_of_the_batch, first_token_second_element_of_the_batch],
             [second_token_first_element_of_the_batch, second_token_second_element_of_the_batch]],
            device=self.a_device)

        actual = nll_loss(predict_tensor, self.a_short_ground_truth).detach().tolist()
        expected = torch.tensor(1, device=self.a_device).tolist()
        self.assertAlmostEqual(expected, actual, delta=5)

    def test_givenACompletelyWrongPredictionTensor_whenNLLLossPerTag_thenLossIsMinus2(self):
        # Tags prediction value are 'inverted' to mimic the log
        first_token_first_element_of_the_batch = [1., 0.]  # the predicted token is the second class
        first_token_second_element_of_the_batch = [0., 1.]  # the predicted token is the first class
        second_token_first_element_of_the_batch = [0., 1.]  # the predicted token is the first class
        second_token_second_element_of_the_batch = [1., 0.]  # the predicted token is the second class
        predict_tensor = torch.tensor(
            [[first_token_first_element_of_the_batch, first_token_second_element_of_the_batch],
             [second_token_first_element_of_the_batch, second_token_second_element_of_the_batch]],
            device=self.a_device)

        actual = nll_loss(predict_tensor, self.a_short_ground_truth).detach().tolist()
        expected = torch.tensor(0., device=self.a_device)
        self.assertAlmostEqual(expected, actual, delta=5)


if __name__ == "__main__":
    unittest.main()
