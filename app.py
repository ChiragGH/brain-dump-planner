import os
from datetime import date
from pathlib import Path
import json
import re

from flask import Flask, jsonify, request, send_from_directory
import requests


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"

# Paste your DeepSeek key here before running the app.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
GOOGLE_TASKS_SCOPES = ["https://www.googleapis.com/auth/tasks"]

SYSTEM_PROMPT = (
    "You are an elite productivity assistant. The user will provide an unstructured text dump "
    "of their daily thoughts. Your job is to extract actionable tasks and return a strictly "
    "formatted JSON array. "
    "For each task, provide:\n"
    "- title: A concise, actionable title.\n"
    "- notes: A detailed description containing: 1. What needs to be done. 2. A brief, logical "
    "approach on how to do it. 3. Estimated time/duration to complete it. "
    "Return ONLY the raw JSON array of objects."
)

VALID_TIME_BLOCKS = {"Morning", "Afternoon", "Evening"}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _import_google_tasks_dependencies():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Google Tasks dependencies are missing. Install: pip install "
            "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    return Request, Credentials, InstalledAppFlow, build


def authenticate_google_tasks():
    Request, Credentials, InstalledAppFlow, build = _import_google_tasks_dependencies()

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), GOOGLE_TASKS_SCOPES)
        except (OSError, ValueError):
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if not CREDENTIALS_FILE.exists():
            raise FileNotFoundError(
                "Missing credentials.json for Google Tasks OAuth. Place your Google OAuth client "
                "file at: "
                + str(CREDENTIALS_FILE)
            )

        flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), GOOGLE_TASKS_SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")

        try:
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Could not write token.json: {exc}") from exc

    return build("tasks", "v1", credentials=creds, cache_discovery=False)


def get_or_create_daily_task_list(service):
    today = date.today()
    list_title = f"{today.strftime('%B')} {today.day}, {today.year}"

    page_token = None
    while True:
        response = service.tasklists().list(maxResults=100, pageToken=page_token).execute()
        for task_list in response.get("items", []):
            title = str(task_list.get("title") or "").strip()
            if title == list_title:
                task_list_id = task_list.get("id")
                if not task_list_id:
                    raise ValueError("Found matching Google Task list but it has no id.")
                return task_list_id, list_title

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    created = service.tasklists().insert(body={"title": list_title}).execute()
    task_list_id = created.get("id")
    if not task_list_id:
        raise ValueError("Google Task list was created but id is missing in API response.")

    return task_list_id, list_title


def _extract_json_payload(raw_text):
    if not raw_text:
        raise ValueError("Empty response from model")

    cleaned = raw_text.strip()

    # Strip optional markdown code fences if the model returns them.
    fence_char = chr(96)
    fence = fence_char * 3
    if cleaned.startswith(fence):
        fence_pattern = r"^" + re.escape(fence) + r"(?:json)?"
        cleaned = re.sub(fence_pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.endswith(fence):
            cleaned = cleaned[:-len(fence)].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        return json.loads(object_match.group(0))

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        return json.loads(array_match.group(0))

    raise ValueError("Model output did not contain valid JSON")


def _normalize_time_of_day(value):
    raw = str(value or "").strip().lower()
    if "morn" in raw:
        return "Morning"
    if "after" in raw or "noon" in raw:
        return "Afternoon"
    if "even" in raw or "night" in raw:
        return "Evening"
    return "Afternoon"


def _infer_time_of_day(title, notes):
    combined = (str(title or "") + " " + str(notes or "")).lower()

    if any(keyword in combined for keyword in ["morning", "subah", "workout", "gym", "breakfast"]):
        return "Morning"
    if any(keyword in combined for keyword in ["evening", "night", "shaam", "dinner"]):
        return "Evening"
    return "Afternoon"


def _ensure_rich_notes(title, notes):
    cleaned_notes = str(notes or "").strip()
    if cleaned_notes:
        return cleaned_notes

    return (
        "What to do: "
        + str(title)
        + ".\n"
        + "How to do it: Break this into clear sub-steps and execute them in order.\n"
        + "Estimated time required: 30-60 minutes."
    )


def _normalize_tasks(data):
    if isinstance(data, dict):
        source = data.get("tasks", [])
    elif isinstance(data, list):
        source = data
    else:
        source = []

    normalized = []
    for item in source:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or item.get("task") or item.get("name") or "").strip()
        if not title:
            continue

        category = str(item.get("category") or "General").strip() or "General"
        notes = _ensure_rich_notes(
            title,
            item.get("notes") or item.get("description") or item.get("details"),
        )
        inferred_time = _infer_time_of_day(title, notes)
        time_of_day = _normalize_time_of_day(item.get("time_of_day") or item.get("time") or inferred_time)

        if time_of_day not in VALID_TIME_BLOCKS:
            time_of_day = "Afternoon"

        normalized.append(
            {
                "title": title,
                "category": category,
                "time_of_day": time_of_day,
                "notes": notes,
            }
        )

    return normalized


def _sync_tasks_to_google_task_list(service, task_list_id, tasks):
    failures = []

    for task in tasks:
        body = {"title": task["title"]}
        notes = str(task.get("notes") or "").strip()
        if notes:
            body["notes"] = notes

        try:
            service.tasks().insert(tasklist=task_list_id, body=body).execute()
        except Exception as exc:
            failures.append({"title": task["title"], "error": str(exc)})

    synced_count = len(tasks) - len(failures)
    return synced_count, failures


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/organize", methods=["POST", "OPTIONS"])
def organize_day():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    raw_text = str(payload.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "Please enter your brain dump before submitting."}), 400

    if not DEEPSEEK_API_KEY:
        return (
            jsonify(
                {
                    "error": (
                        "DeepSeek API key not set. Set DEEPSEEK_API_KEY in your environment."
                    )
                }
            ),
            500,
        )

    user_prompt = f"""
User Brain Dump:
{raw_text}

Return ONLY a raw JSON array (no markdown) in this format:
[
    {{
        "title": "string",
        "notes": "What to do: ... How to do it: ... Estimated time required: ..."
    }}
]

Rules:
- Extract only actionable tasks.
- Keep each title concise and executable.
- Notes must include: what to do, how to do it, and estimated duration.
- You may optionally include "category" and "time_of_day".
""".strip()

    response = None
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            timeout=75,
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"Failed to reach DeepSeek API: {exc}"}), 502

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json()
        except ValueError:
            pass
        return jsonify({"error": "DeepSeek API returned an error.", "details": detail}), response.status_code

    try:
        model_output = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return jsonify({"error": "Unexpected response format from DeepSeek API."}), 502

    try:
        parsed = _extract_json_payload(model_output)
        tasks = _normalize_tasks(parsed)
    except (ValueError, json.JSONDecodeError) as exc:
        return (
            jsonify(
                {
                    "error": "Could not parse DeepSeek response into valid task JSON.",
                    "details": str(exc),
                    "raw_response": model_output,
                }
            ),
            502,
        )

    if not tasks:
        return (
            jsonify(
                {
                    "error": "No actionable tasks were extracted. Try a more detailed dump.",
                    "raw_response": model_output,
                }
            ),
            422,
        )

    try:
        google_service = authenticate_google_tasks()
        task_list_id, task_list_title = get_or_create_daily_task_list(google_service)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": "Failed to initialize Google Tasks API.", "details": str(exc)}), 502

    synced_count, sync_failures = _sync_tasks_to_google_task_list(google_service, task_list_id, tasks)
    if sync_failures:
        return (
            jsonify(
                {
                    "error": "One or more tasks could not be synced to Google Tasks.",
                    "task_list_title": task_list_title,
                    "synced_count": synced_count,
                    "failed": sync_failures,
                }
            ),
            502,
        )

    summary = {
        "total": len(tasks),
        "morning": sum(1 for t in tasks if t["time_of_day"] == "Morning"),
        "afternoon": sum(1 for t in tasks if t["time_of_day"] == "Afternoon"),
        "evening": sum(1 for t in tasks if t["time_of_day"] == "Evening"),
    }

    sync_message = (
        f"Successfully synced {synced_count} task(s) to Google Tasks list '{task_list_title}'."
    )

    return jsonify(
        {
            "tasks": tasks,
            "summary": summary,
            "synced_count": synced_count,
            "task_list_title": task_list_title,
            "sync_message": sync_message,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
