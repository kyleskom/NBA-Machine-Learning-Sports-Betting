import unittest
from src.Utils.Kelly_Criterion import calculate_kelly_criterion, american_to_decimal

class TestKellyCriterion(unittest.TestCase):


    def test_american_to_decimal_positive(self):
        self.assertEqual(american_to_decimal(150), 2.5)

    def test_american_to_decimal_negative(self):
        self.assertEqual(american_to_decimal(-200), 1.5)

    def test_american_to_decimal_negative_large(self):
        self.assertEqual(american_to_decimal(-500), 1.2)

    def test_calculate_kelly_criterion(self):
        result = calculate_kelly_criterion(150, 0.6)
        print(f"Result: {result}")
        self.assertNotEqual(result, 0)

    def test_calculate_kelly_criterion_negative_odds(self):
        result = calculate_kelly_criterion(-200, 0.7)
        print(f"Result: {result}")
        self.assertNotEqual(result, 0)

    def test_calculate_kelly_criterion_negative_odds_large(self):
        result = calculate_kelly_criterion(-500, 0.8)
        print(f"Result: {result}")
        self.assertNotEqual(result, 0)


    def test_calculate_kelly_criterion_non_zero(self):
        result = calculate_kelly_criterion(150, 0.6)
        print(f"Result: {result}")
        self.assertNotEqual(result, 0)

if __name__ == '__main__':
    unittest.main()
