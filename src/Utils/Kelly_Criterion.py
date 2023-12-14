def american_to_decimal(american_odds):
    """
    Converts American odds to decimal odds (European odds).
    """
    if american_odds >= 100:
        decimal_odds = 1 + (american_odds / 100)
    else:
        decimal_odds = (100 / abs(american_odds)) + 1
    return round(decimal_odds, 2)

def calculate_kelly_criterion(american_odds, model_prob):
    """
    Calculates the fraction of the bankroll to be wagered on each bet using the Kelly Criterion.
    """
    decimal_odds = american_to_decimal(american_odds)
    print(decimal_odds)
    p = model_prob
    q = 1 - p
    b = decimal_odds / 100
    kelly_fraction = (p * b - q) / b
    return round(kelly_fraction, 2) if kelly_fraction > 0 else 0