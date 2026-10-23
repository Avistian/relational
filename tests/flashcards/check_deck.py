"""Read-only validation of evaluated PAPER_DECK cards."""

REQUIRED_FIELDS = ("id", "paper", "year", "lesson", "front", "back")


def validate_deck(cards, report_total):
    """Raise ValueError for missing fields, duplicate IDs or report count drift."""
    seen = set()
    for index, card in enumerate(cards):
        for field in REQUIRED_FIELDS:
            value = card.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"required field {field} missing or empty at card {index}")
        card_id = card["id"]
        if not isinstance(card_id, str):
            raise ValueError(f"required field id must be a string at card {index}")
        if card_id in seen:
            raise ValueError(f"duplicate card id: {card_id}")
        seen.add(card_id)
    if len(cards) != report_total:
        raise ValueError(f"deck count {len(cards)} differs from report count {report_total}")
