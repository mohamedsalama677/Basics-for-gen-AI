"""Function tools exposed to the LLM.

Docstrings are load-bearing — LiveKit passes them to the LLM as the tool's
description, so the LLM uses them to decide when and how to call the tool.
"""

import logging
import re

from livekit.agents import function_tool

log = logging.getLogger("swifteats.tools")

# Fake order database — stand-in for what would be a real backend call.
MOCK_ORDERS: dict[str, dict] = {
    "5473": {
        "status": "out for delivery",
        "eta_min": 12,
        "items": ["Margherita pizza"],
    },
    "9911": {
        "status": "preparing",
        "eta_min": 30,
        "items": ["Chicken shawarma", "Pepsi"],
    },
    "0001": {
        "status": "delivered",
        "eta_min": 0,
        "items": ["Falafel wrap"],
    },
}

ORDER_ID_PATTERN = re.compile(r"^\d{4}$")


@function_tool
async def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer's food-delivery order.

    Use this whenever the customer asks about "my order", "where's my food",
    or references a numeric order ID. If the customer hasn't given you an
    order ID yet, ask them for it before calling this tool.

    Args:
        order_id: The 4-digit order number the customer received in their
            confirmation SMS or email. Must be exactly 4 digits.

    Returns:
        A short human-readable status the assistant can speak aloud.
    """
    log.info("[tool] get_order_status called with order_id=%r", order_id)

    if not ORDER_ID_PATTERN.match(order_id):
        return (
            f"That doesn't look like a valid order ID. "
            f"Order IDs are exactly 4 digits, but I got {order_id!r}."
        )

    order = MOCK_ORDERS.get(order_id)
    if not order:
        return (
            f"I couldn't find any order with ID {order_id}. "
            f"Could you double-check the number?"
        )

    return (
        f"Order {order_id}: {order['status']}. "
        f"ETA {order['eta_min']} minutes. "
        f"Items: {', '.join(order['items'])}."
    )
