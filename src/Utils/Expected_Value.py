def expected_value(Pwin, odds):
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100

def k_odds(american_odds):
    if american_odds > 0:
        return (american_odds/100)
    else:
        return -1 * (100/american_odds)

def kelly(Pwin, american_odds, bankroll=100, adjustment=1.0):
    odds = k_odds(american_odds)
    f = Pwin - ((1 - Pwin)/odds)
    wager = (f * bankroll) / adjustment
    wager = 0 if wager < 0 else wager
    return round(wager,2)