from schemas import PlayCard
from config import THRESHOLDS

def enforce(card: PlayCard) -> PlayCard:
    # Add/adjust preconditions if spread or OI borderline
    if THRESHOLDS["rr_min"] and card.risk_reward < THRESHOLDS["rr_min"]:
        card.notes += " | Rejected: RR below threshold."
    return card
