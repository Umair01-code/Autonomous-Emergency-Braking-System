def assess_risk(ttc):
    """Classify collision risk using TTC in seconds."""

    if ttc == float("inf"):
        return "SAFE"

    if ttc > 3.0:
        return "SAFE"

    if ttc > 1.5:
        return "WARNING"

    return "AEB"
