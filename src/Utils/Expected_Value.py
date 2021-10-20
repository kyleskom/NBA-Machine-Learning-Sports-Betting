def expected_value(Pwin, odds):
    Ploss = 1 - Pwin
    Mwin = payout(odds)
    return round((Pwin * Mwin) - (Ploss * 100), 2)


def payout(odds):
    if odds > 0:
        return odds
    else:
        return (100 / (-1 * odds)) * 100
