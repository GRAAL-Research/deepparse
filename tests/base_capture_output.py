import io
import sys
from unittest import TestCase


class CaptureOutputTestCase(TestCase):

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out
