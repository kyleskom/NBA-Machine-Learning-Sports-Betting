import unittest
from src.Utils import Expected_Value


class TestExpectedValue(unittest.TestCase):

    def test_expected_value_1(self):
        result = Expected_Value.expected_value(.76, -200)
        self.assertEqual(result, 14)

    def test_expected_value_2(self):
        result = Expected_Value.expected_value(.3, -500)
        self.assertEqual(result, -64)

    def test_expected_value_3(self):
        result = Expected_Value.expected_value(.6, 250)
        self.assertEqual(result, 110)

    def test_expected_value_4(self):
        result = Expected_Value.expected_value(.2, -200)
        self.assertEqual(result, -70)

    def test_expected_value_5(self):
        result = Expected_Value.expected_value(.8137, -200)
        self.assertEqual(result, 22.05)

    def test_expected_value_6(self):
        result = Expected_Value.expected_value(.2175, -550)
        self.assertEqual(result, -74.30)

    def test_expected_value_7(self):
        result = Expected_Value.expected_value(.5298, 1000)
        self.assertEqual(result, 482.78)

    def test_expected_value_8(self):
        result = Expected_Value.expected_value(.638, 275)
        self.assertEqual(result, 139.25)
