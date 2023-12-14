import unittest
from src.Utils.Kelly_Criterion import calculate_kelly_criterion, american_to_decimal

class TestKellyCriterion(unittest.TestCase):

    def test_calculate_kelly_criterion_1(self):
        result = calculate_kelly_criterion(-110, 0.6)
        self.assertEqual(result, 0)

    def test_calculate_kelly_criterion_2(self):
        result = calculate_kelly_criterion(-110, 0.4)
        self.assertEqual(result, 0)

    def test_calculate_kelly_criterion_3(self):
        result = calculate_kelly_criterion(400, 0.35)
        self.assertEqual(result, 0)

    def test_calculate_kelly_criterion_4(self):
        result = calculate_kelly_criterion(-500, 0.85)
        self.assertEqual(result, 0)

    def test_calculate_kelly_criterion_5(self):
        result = calculate_kelly_criterion(100, 0.99)
        self.assertEqual(result, 0.49)

    def test_american_to_decimal_positive(self):
        result = american_to_decimal(120)
        self.assertEqual(result, 2.2)

    def test_american_to_decimal_negative(self):
        result = american_to_decimal(-150)
        self.assertEqual(result, 1.67)

if __name__ == '__main__':
    unittest.main()
