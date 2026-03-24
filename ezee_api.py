"""
eZee PMS API integration module.
Handles room availability checks and booking creation via the eZee Reservation API.

API docs: https://api.ezeetechnosys.com/
Base URL: https://live.ipms247.com/booking/reservation_api/listing.php
"""

import os
import logging
from datetime import datetime

import aiohttp

logger = logging.getLogger(__name__)

# eZee API configuration
EZEE_BASE_URL = os.environ.get(
    "EZEE_BASE_URL",
    "https://live.ipms247.com/booking/reservation_api/listing.php",
)
EZEE_HOTEL_CODE = os.environ.get("EZEE_HOTEL_CODE", "")
EZEE_AUTH_CODE = os.environ.get("EZEE_AUTH_CODE", "")
EZEE_API_KEY = os.environ.get("EZEE_API_KEY", "")


def _auth_block() -> dict:
    """Return the standard eZee authentication block."""
    return {"HotelCode": EZEE_HOTEL_CODE, "AuthCode": EZEE_AUTH_CODE}


def is_configured() -> bool:
    """Check whether eZee API credentials are present."""
    return bool(EZEE_HOTEL_CODE and EZEE_AUTH_CODE and EZEE_API_KEY)


async def check_availability(
    check_in: str,
    check_out: str,
    num_rooms: int = 1,
    num_adults: int = 1,
    room_type_id: str | None = None,
) -> dict:
    """
    Check room availability for the given date range.

    Args:
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        num_rooms: Number of rooms requested
        num_adults: Number of adult guests
        room_type_id: Optional specific room type ID

    Returns:
        dict with availability results or error info
    """
    if not is_configured():
        return {"error": "eZee API credentials not configured"}

    request_data = {
        "RES_Request": {
            "Request_Type": "RoomAvailability",
            "Authentication": _auth_block(),
            "FromDate": check_in,
            "ToDate": check_out,
            "NightCount": _night_count(check_in, check_out),
            "NoOfRooms": str(num_rooms),
            "NoOfAdults": str(num_adults),
        }
    }

    if room_type_id:
        request_data["RES_Request"]["RoomTypeID"] = room_type_id

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                EZEE_BASE_URL,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                result = await resp.json(content_type=None)
                logger.info(f"eZee availability response: {resp.status}")

                if resp.status != 200:
                    return {"error": f"eZee API returned status {resp.status}"}

                return _parse_availability(result)

    except aiohttp.ClientError as e:
        logger.error(f"eZee API request failed: {e}")
        return {"error": f"Could not reach eZee API: {e}"}
    except Exception as e:
        logger.error(f"eZee availability check error: {e}", exc_info=True)
        return {"error": str(e)}


async def create_booking(
    check_in: str,
    check_out: str,
    guest_name: str,
    guest_email: str = "",
    guest_phone: str = "",
    room_type_id: str = "",
    num_rooms: int = 1,
    num_adults: int = 1,
    special_requests: str = "",
) -> dict:
    """
    Create a new reservation in eZee PMS.

    Args:
        check_in: Check-in date (YYYY-MM-DD)
        check_out: Check-out date (YYYY-MM-DD)
        guest_name: Full name of the guest
        guest_email: Guest email address
        guest_phone: Guest phone number
        room_type_id: Room type ID to book
        num_rooms: Number of rooms
        num_adults: Number of adults
        special_requests: Any special requests from the guest

    Returns:
        dict with booking confirmation or error info
    """
    if not is_configured():
        return {"error": "eZee API credentials not configured"}

    # Split guest name into first/last
    name_parts = guest_name.strip().split(maxsplit=1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    request_data = {
        "RES_Request": {
            "Request_Type": "InsertBooking",
            "Authentication": _auth_block(),
            "Booking": {
                "CheckInDate": check_in,
                "CheckOutDate": check_out,
                "NightCount": _night_count(check_in, check_out),
                "NoOfRooms": str(num_rooms),
                "NoOfAdults": str(num_adults),
                "Salutation": "",
                "FirstName": first_name,
                "LastName": last_name,
                "Email": guest_email,
                "PhoneNo": guest_phone,
                "RoomTypeID": room_type_id,
                "SpecialRequest": special_requests,
                "Source": "VoiceAgent",
            },
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                EZEE_BASE_URL,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                result = await resp.json(content_type=None)
                logger.info(f"eZee booking response: {resp.status}")

                if resp.status != 200:
                    return {"error": f"eZee API returned status {resp.status}"}

                return _parse_booking(result)

    except aiohttp.ClientError as e:
        logger.error(f"eZee API request failed: {e}")
        return {"error": f"Could not reach eZee API: {e}"}
    except Exception as e:
        logger.error(f"eZee booking creation error: {e}", exc_info=True)
        return {"error": str(e)}


# --- Helpers ---


def _night_count(check_in: str, check_out: str) -> str:
    """Calculate the number of nights between two dates."""
    try:
        d_in = datetime.strptime(check_in, "%Y-%m-%d")
        d_out = datetime.strptime(check_out, "%Y-%m-%d")
        return str(max((d_out - d_in).days, 1))
    except ValueError:
        return "1"


def _parse_availability(data: dict) -> dict:
    """Parse the eZee availability response into a clean format."""
    # eZee wraps responses in various structures; handle common patterns
    if isinstance(data, dict) and "Errors" in data:
        return {"error": data["Errors"]}

    rooms = []
    # Try to extract room data from known response structures
    room_data = data.get("RoomTypes", data.get("Rooms", []))
    if isinstance(room_data, dict):
        room_data = list(room_data.values())

    for room in room_data if isinstance(room_data, list) else []:
        rooms.append({
            "room_type_id": room.get("RoomTypeID", ""),
            "room_type_name": room.get("RoomTypeName", room.get("Name", "")),
            "available_count": room.get("Availability", room.get("AvailableRooms", 0)),
            "rate": room.get("Rate", room.get("Price", "")),
            "description": room.get("Description", ""),
        })

    if not rooms and isinstance(data, dict):
        # Return raw data if we couldn't parse it into our structure
        return {"available": True, "raw_data": data}

    return {"available": len(rooms) > 0, "rooms": rooms}


def _parse_booking(data: dict) -> dict:
    """Parse the eZee booking response into a clean format."""
    if isinstance(data, dict) and "Errors" in data:
        return {"error": data["Errors"]}

    reservation_no = data.get("ReservationNo", data.get("BookingId", ""))
    status = data.get("Status", data.get("BookingStatus", ""))

    if reservation_no:
        return {
            "success": True,
            "reservation_no": reservation_no,
            "status": status or "Confirmed",
        }

    # Return raw data if we couldn't find a reservation number
    return {"success": False, "raw_data": data}
