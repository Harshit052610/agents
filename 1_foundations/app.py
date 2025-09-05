import os
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr

# --- Load .env and optional Gemini key ---
load_dotenv(override=True)
gemini_key = os.getenv("GOOGLE_API_KEY") 
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        print("Gemini API key configured from environment.")
    except Exception as ex:
        print("Warning: couldn't configure genai with provided key:", ex)
else:
    print("Warning: No Gemini API key found. Set GEMINI_API_KEY or GENAI_API_KEY if required.")

# --- Pushover Setup (optional) ---
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

if pushover_user:
    print(f"Pushover user found and starts with {pushover_user[0]}")
else:
    print("Pushover user not found")

if pushover_token:
    print(f"Pushover token found and starts with {pushover_token[0]}")
else:
    print("Pushover token not found")


def push(message):
    """Sends a message to Pushover (best-effort)."""
    print(f"Push: {message}")
    if not (pushover_user and pushover_token):
        print("Pushover credentials missing — skipping HTTP push.")
        return
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    try:
        requests.post(pushover_url, data=payload, timeout=5)
    except Exception as e:
        print(f"Pushover request failed: {e}")


def record_user_details(email, name="Name not provided", notes="not provided"):
    """Records user's contact details via Pushover (manual call)."""
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    """Records a question that the LLM couldn't answer (manual call)."""
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


# --- Document Loading ---
try:
    reader = PdfReader("me/DA_RESUME.pdf")
    linkedin = "".join(page.extract_text() or "" for page in reader.pages)
except FileNotFoundError:
    linkedin = "Could not find LinkedIn file."
    print("Warning: LinkedIn file 'me/DA_RESUME.pdf' not found. Using a placeholder.")

try:
    with open("me/summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()
except FileNotFoundError:
    summary = "Could not find summary file."
    print("Warning: Summary file 'me/summary.txt' not found. Using a placeholder.")


# --- System prompt moved into first 'user' message (Gemini-friendly) ---
name = "N Harshit"
system_prompt = f"""[SYSTEM INSTRUCTION — for the assistant to follow]
You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Represent {name} faithfully and professionally. If you don't know an answer, suggest the user share more details or provide contact information.
If the user asks to get in touch, ask for their email.

## Summary:
{summary}

## LinkedIn Profile:
{linkedin}
"""


# --- Robust content extractor for multiple SDK variants ---
def extract_text_from_response(response_obj):
    """
    Return best-effort text from genai.generate_content response object.
    Works across SDK variants that use .candidates[0].content.text or .parts.
    """
    try:
        if not getattr(response_obj, "candidates", None):
            return str(response_obj)

        cand = response_obj.candidates[0]
        content = getattr(cand, "content", None) or cand

        # Many SDKs put full text at content.text
        text = getattr(content, "text", None)
        if text:
            return text

        # Some SDKs return parts with 'text' fields
        parts = getattr(content, "parts", None)
        if parts:
            assembled = ""
            for p in parts:
                if isinstance(p, dict):
                    t = p.get("text") or p.get("content") or ""
                else:
                    t = getattr(p, "text", None) or getattr(p, "content", None) or ""
                if t:
                    assembled += t
            if assembled:
                return assembled

        # Fallback
        return str(cand)

    except Exception as e:
        return f"[Could not extract text from response: {e}]"


# --- Normalize Gradio history (handles many shapes) ---
def normalize_history(raw_history):
    """
    Normalize Gradio history entries into a list of (user_msg, model_msg) tuples.
    Handles:
      - [(user, model), ...]
      - [[user, model, extra], ...]
      - [{'user':..., 'assistant':...}, ...]
      - [{'role':..., 'content':...}, ...] (best-effort)
      - single strings
    """
    normalized = []
    for item in raw_history or []:
        user_msg = ""
        model_msg = ""

        # lists/tuples: take first two elements
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                user_msg = "" if item[0] is None else str(item[0])
                model_msg = "" if item[1] is None else str(item[1])
            elif len(item) == 1:
                user_msg = "" if item[0] is None else str(item[0])

        # dicts: try common fields
        elif isinstance(item, dict):
            # common Gradio shape: {'message': 'text', 'sender': 'user' } but not consistent
            if "user" in item and "assistant" in item:
                user_msg = "" if item.get("user") is None else str(item.get("user"))
                model_msg = "" if item.get("assistant") is None else str(item.get("assistant"))
            elif "sender" in item and "message" in item:
                if item.get("sender") == "user":
                    user_msg = "" if item.get("message") is None else str(item.get("message"))
                else:
                    model_msg = "" if item.get("message") is None else str(item.get("message"))
            elif "role" in item and "content" in item:
                if item.get("role") == "user":
                    user_msg = "" if item.get("content") is None else str(item.get("content"))
                elif item.get("role") in ("assistant", "model"):
                    model_msg = "" if item.get("content") is None else str(item.get("content"))
            else:
                # fallback: take first two values in the dict
                vals = list(item.values())
                if len(vals) >= 2:
                    user_msg = "" if vals[0] is None else str(vals[0])
                    model_msg = "" if vals[1] is None else str(vals[1])
                elif len(vals) == 1:
                    user_msg = "" if vals[0] is None else str(vals[0])

        else:
            # anything else: stringify to user message
            user_msg = str(item)

        normalized.append((user_msg, model_msg))
    return normalized


# --- Chat function used by Gradio ---
def chat(message, history):
    """
    message: str
    history: list (various shapes) from Gradio ChatInterface
    """
    # Put the system instructions as the first user message so Gemini sees them
    full_history = [{"role": "user", "parts": [{"text": system_prompt}]}]

    # Normalize and append prior conversation
    for user_msg, model_msg in normalize_history(history):
        if user_msg:
            full_history.append({"role": "user", "parts": [{"text": user_msg}]})
        if model_msg:
            full_history.append({"role": "model", "parts": [{"text": model_msg}]})

    # Add current user message
    full_history.append({"role": "user", "parts": [{"text": message}]})

    # Create the model instance (no tools declared to avoid proto issues)
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        response = model.generate_content(full_history)
        answer = extract_text_from_response(response)
        if not answer or answer.strip() == "":
            return f"(no text extracted) {response}"
        return answer

    except Exception as e:
        return f"An error occurred: {e}"


# --- Launch Gradio Interface ---
if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()
