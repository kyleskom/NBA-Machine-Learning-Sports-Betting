import unittest
from src.Utils import Kelly_Criterion as kc


class TestKellyCriterion(unittest.TestCase):

    def test_calculate_kelly_criterion_1(self):
        result = kc.calculate_kelly_criterion(-110, .6)
        self.assertEqual(result, 16.04)

    def test_calculate_kelly_criterion_2(self):
        result = kc.calculate_kelly_criterion(-110, .4)
        self.assertEqual(result, 0)

    def test_calculate_kelly_criterion_3(self):
        result = kc.calculate_kelly_criterion(400, .35)
        self.assertEqual(result, 18.75)

    def test_calculate_kelly_criterion_4(self):
        result = kc.calculate_kelly_criterion(-500, .85)
        self.assertEqual(result, 10)

    def test_calculate_kelly_criterion_5(self):
        result = kc.calculate_kelly_criterion(100, .99)
        self.assertEqual(result, 98)
