"""
Claude tool definitions and execution for the Thiva voice agent.
Provides hotel booking tools that Claude can call during conversation.
"""

import json
import logging

from ezee_api import check_availability, create_booking, is_configured

logger = logging.getLogger(__name__)

# Tool definitions sent to Claude API
TOOL_DEFINITIONS = [
    {
        "name": "check_availability",
        "description": (
            "Check room availability at the hotel for a given date range. "
            "Use this when a guest asks about available rooms, rates, or whether "
            "they can stay on specific dates. Returns available room types with rates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "check_in": {
                    "type": "string",
                    "description": "Check-in date in YYYY-MM-DD format",
                },
                "check_out": {
                    "type": "string",
                    "description": "Check-out date in YYYY-MM-DD format",
                },
                "num_rooms": {
                    "type": "integer",
                    "description": "Number of rooms needed (default 1)",
                    "default": 1,
                },
                "num_adults": {
                    "type": "integer",
                    "description": "Number of adult guests (default 1)",
                    "default": 1,
                },
            },
            "required": ["check_in", "check_out"],
        },
    },
    {
        "name": "create_booking",
        "description": (
            "Create a new hotel reservation. Use this ONLY after confirming "
            "availability and getting explicit confirmation from the guest. "
            "Requires guest name and dates at minimum."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "check_in": {
                    "type": "string",
                    "description": "Check-in date in YYYY-MM-DD format",
                },
                "check_out": {
                    "type": "string",
                    "description": "Check-out date in YYYY-MM-DD format",
                },
                "guest_name": {
                    "type": "string",
                    "description": "Full name of the guest",
                },
                "guest_phone": {
                    "type": "string",
                    "description": "Guest phone number",
                    "default": "",
                },
                "guest_email": {
                    "type": "string",
                    "description": "Guest email address",
                    "default": "",
                },
                "room_type_id": {
                    "type": "string",
                    "description": "Room type ID from availability check",
                    "default": "",
                },
                "num_rooms": {
                    "type": "integer",
                    "description": "Number of rooms to book",
                    "default": 1,
                },
                "num_adults": {
                    "type": "integer",
                    "description": "Number of adult guests",
                    "default": 1,
                },
                "special_requests": {
                    "type": "string",
                    "description": "Any special requests from the guest",
                    "default": "",
                },
            },
            "required": ["check_in", "check_out", "guest_name"],
        },
    },
]


def get_tools() -> list[dict]:
    """Return tool definitions if eZee is configured, else empty list."""
    if is_configured():
        return TOOL_DEFINITIONS
    logger.warning("eZee not configured — tools disabled")
    return []


async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool call from Claude and return the result as a string.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters from Claude

    Returns:
        JSON string with the tool result
    """
    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")

    if tool_name == "check_availability":
        result = await check_availability(
            check_in=tool_input["check_in"],
            check_out=tool_input["check_out"],
            num_rooms=tool_input.get("num_rooms", 1),
            num_adults=tool_input.get("num_adults", 1),
        )
    elif tool_name == "create_booking":
        result = await create_booking(
            check_in=tool_input["check_in"],
            check_out=tool_input["check_out"],
            guest_name=tool_input["guest_name"],
            guest_email=tool_input.get("guest_email", ""),
            guest_phone=tool_input.get("guest_phone", ""),
            room_type_id=tool_input.get("room_type_id", ""),
            num_rooms=tool_input.get("num_rooms", 1),
            num_adults=tool_input.get("num_adults", 1),
            special_requests=tool_input.get("special_requests", ""),
        )
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    logger.info(f"Tool result: {result}")
    return json.dumps(result)
