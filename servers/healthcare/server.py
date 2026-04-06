#!/usr/bin/env python3
import json
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any

MCP_PROTOCOL_VERSION = "2024-11-05"

patients_db: dict[str, dict[str, Any]] = {}
appointments_db: dict[str, dict[str, Any]] = {}
session_data: dict[str, dict[str, Any]] = {"default": {"patient_fields": {}, "appointment_fields": {}}}

DOCTORS = [
    {"id": "DOC001", "name": "Dr. Sarah Chen", "specialty": "general"},
    {"id": "DOC002", "name": "Dr. Michael Patel", "specialty": "cardiology"},
    {"id": "DOC003", "name": "Dr. Jennifer Lee", "specialty": "pediatrics"},
]

LOCATIONS = [
    {"id": "LOC001", "name": "Downtown Medical Center", "address": "123 Main St"},
    {"id": "LOC002", "name": "Westside Clinic", "address": "456 Oak Ave"},
]


def store_patient_field(field_name: str, field_value: str, session_id: str = "default") -> dict[str, Any]:
    if session_id not in session_data:
        session_data[session_id] = {"patient_fields": {}, "appointment_fields": {}}
    session_data[session_id]["patient_fields"][field_name] = field_value
    return {"success": True, "field": field_name, "value": field_value}


def get_patient_fields(session_id: str = "default") -> dict[str, Any]:
    fields = session_data.get(session_id, {}).get("patient_fields", {})
    required = ["first_name", "last_name", "date_of_birth", "phone", "email"]
    missing = [f for f in required if f not in fields]
    return {"fields": fields, "count": len(fields), "missing_required": missing}


def create_patient(
    first_name: str,
    last_name: str,
    phone: str,
    email: str = "",
    date_of_birth: str = "",
    insurance_provider: str = "",
    insurance_member_id: str = "",
) -> dict[str, Any]:
    for patient in patients_db.values():
        if patient.get("phone") == phone:
            return {"matched": True, "patient": patient, "message": "Found existing patient"}
        if email and patient.get("email") == email:
            return {"matched": True, "patient": patient, "message": "Found existing patient"}

    patient_id = f"PT{uuid.uuid4().hex[:8].upper()}"
    patient = {
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "phone": phone,
        "email": email,
        "date_of_birth": date_of_birth,
        "insurance_provider": insurance_provider,
        "insurance_member_id": insurance_member_id,
        "created_at": datetime.now().isoformat(),
    }
    patients_db[patient_id] = patient
    return {"matched": False, "patient": patient, "message": "Created new patient"}


def search_patients(phone: str = "", email: str = "", name: str = "") -> dict[str, Any]:
    results = []
    for patient in patients_db.values():
        if phone and patient.get("phone") == phone:
            results.append(patient)
        elif email and patient.get("email") == email:
            results.append(patient)
        elif name and name.lower() in f"{patient['first_name']} {patient['last_name']}".lower():
            results.append(patient)
    return {"patients": results, "count": len(results)}


def get_available_appointments(
    start_date: str = "",
    end_date: str = "",
    doctor_id: str = "",
    specialty: str = "",
) -> dict[str, Any]:
    start = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now()
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else start + timedelta(days=7)

    slots = []
    current = start
    slot_num = 1

    while current <= end and len(slots) < 15:
        if current.weekday() < 5:
            for hour in [9, 10, 11, 14, 15, 16]:
                for doc in DOCTORS:
                    if doctor_id and doc["id"] != doctor_id:
                        continue
                    if specialty and doc["specialty"] != specialty:
                        continue

                    for loc in LOCATIONS:
                        slots.append({
                            "slot_id": f"SLOT{slot_num:05d}",
                            "doctor_id": doc["id"],
                            "doctor_name": doc["name"],
                            "specialty": doc["specialty"],
                            "location_id": loc["id"],
                            "location_name": loc["name"],
                            "date": current.strftime("%Y-%m-%d"),
                            "time": f"{hour}:00",
                            "display": f"{current.strftime('%A, %B %d')} at {hour}:00 with {doc['name']}",
                        })
                        slot_num += 1
        current += timedelta(days=1)

    return {"slots": slots[:15], "count": min(len(slots), 15)}


def book_appointment(
    patient_id: str,
    doctor_id: str,
    location_id: str,
    appointment_date: str,
    appointment_time: str,
    reason: str = "",
) -> dict[str, Any]:
    appointment_id = f"APT{uuid.uuid4().hex[:8].upper()}"
    confirmation = f"CONF{uuid.uuid4().hex[:6].upper()}"

    doc = next((d for d in DOCTORS if d["id"] == doctor_id), None)
    loc = next((l for l in LOCATIONS if l["id"] == location_id), None)

    appointment = {
        "appointment_id": appointment_id,
        "confirmation_number": confirmation,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "doctor_name": doc["name"] if doc else "Unknown",
        "location_id": location_id,
        "location_name": loc["name"] if loc else "Unknown",
        "appointment_date": appointment_date,
        "appointment_time": appointment_time,
        "reason": reason,
        "status": "confirmed",
        "created_at": datetime.now().isoformat(),
    }
    appointments_db[appointment_id] = appointment

    return {
        "success": True,
        "appointment_id": appointment_id,
        "confirmation_number": confirmation,
        "message": f"Appointment confirmed for {appointment_date} at {appointment_time}",
    }


def get_patient_appointments(patient_id: str, status: str = "all") -> dict[str, Any]:
    results = [a for a in appointments_db.values() if a["patient_id"] == patient_id]
    if status == "upcoming":
        results = [a for a in results if a["status"] == "confirmed"]
    elif status == "cancelled":
        results = [a for a in results if a["status"] == "cancelled"]
    return {"appointments": results, "count": len(results)}


def cancel_appointment(appointment_id: str, reason: str = "") -> dict[str, Any]:
    if appointment_id not in appointments_db:
        return {"success": False, "error": "Appointment not found"}

    appointments_db[appointment_id]["status"] = "cancelled"
    appointments_db[appointment_id]["cancellation_reason"] = reason
    return {"success": True, "message": "Appointment cancelled successfully"}


def transfer_call(department: str, reason: str = "") -> dict[str, Any]:
    return {
        "transferred": True,
        "department": department,
        "reason": reason,
        "message": f"Transferring to {department}. Please hold.",
    }


TOOLS = {
    "store_patient_field": {
        "handler": store_patient_field,
        "description": "Store a patient information field collected during intake",
        "schema": {
            "type": "object",
            "properties": {
                "field_name": {"type": "string", "description": "Name of the field (e.g., first_name, last_name, phone, email, date_of_birth, insurance_provider)"},
                "field_value": {"type": "string", "description": "Value of the field"},
                "session_id": {"type": "string", "description": "Session identifier", "default": "default"},
            },
            "required": ["field_name", "field_value"],
        },
    },
    "get_patient_fields": {
        "handler": get_patient_fields,
        "description": "Get all patient fields collected so far and see what's missing",
        "schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session identifier", "default": "default"},
            },
            "required": [],
        },
    },
    "create_patient": {
        "handler": create_patient,
        "description": "Create a new patient record or find existing one by phone/email",
        "schema": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string"},
                "date_of_birth": {"type": "string"},
                "insurance_provider": {"type": "string"},
                "insurance_member_id": {"type": "string"},
            },
            "required": ["first_name", "last_name", "phone"],
        },
    },
    "search_patients": {
        "handler": search_patients,
        "description": "Search for existing patients by phone, email, or name",
        "schema": {
            "type": "object",
            "properties": {
                "phone": {"type": "string"},
                "email": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": [],
        },
    },
    "get_available_appointments": {
        "handler": get_available_appointments,
        "description": "Get available appointment slots for scheduling",
        "schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                "doctor_id": {"type": "string", "description": "Filter by specific doctor"},
                "specialty": {"type": "string", "description": "Filter by specialty (general, cardiology, pediatrics)"},
            },
            "required": [],
        },
    },
    "book_appointment": {
        "handler": book_appointment,
        "description": "Book an appointment for a patient",
        "schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "doctor_id": {"type": "string"},
                "location_id": {"type": "string"},
                "appointment_date": {"type": "string", "description": "Date (YYYY-MM-DD)"},
                "appointment_time": {"type": "string", "description": "Time (HH:MM)"},
                "reason": {"type": "string", "description": "Reason for visit"},
            },
            "required": ["patient_id", "doctor_id", "location_id", "appointment_date", "appointment_time"],
        },
    },
    "get_patient_appointments": {
        "handler": get_patient_appointments,
        "description": "Get appointments for a patient",
        "schema": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string"},
                "status": {"type": "string", "description": "Filter by status: all, upcoming, cancelled"},
            },
            "required": ["patient_id"],
        },
    },
    "cancel_appointment": {
        "handler": cancel_appointment,
        "description": "Cancel an existing appointment",
        "schema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["appointment_id"],
        },
    },
    "transfer_call": {
        "handler": transfer_call,
        "description": "Transfer the call to another department",
        "schema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "description": "Department to transfer to (billing, medical_records, nurse, emergency)"},
                "reason": {"type": "string"},
            },
            "required": ["department"],
        },
    },
}


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "healthcare", "version": "1.0.0"},
            },
        }

    elif method == "notifications/initialized":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    elif method == "tools/list":
        tools_list = [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": info["schema"],
            }
            for name, info in TOOLS.items()
        ]
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools_list}}

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
            }

        try:
            handler = TOOLS[tool_name]["handler"]
            result = handler(**arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                },
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def main() -> None:
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = handle_request(request)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
