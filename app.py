# app.py
import os
import glob
import json
import time
import traceback
import urllib.parse
import asyncio
from typing import Any, Generator, Optional

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import inspect

# Try optional external ask functions (phase_2 / phase_1).
ASK_FUNC = None
try:
    from phase4 import ask as ask_func
    ASK_FUNC = ask_func
    print("[ZULTX] Using phase_4.ask")
except Exception:
    try:
        from phase_3 import ask as ask_func
        ASK_FUNC = ask_func
        print("[ZULTX] Using phase_3.ask")
    except Exception as e:
        print("[ZULTX] No phase_3/phase_4 ask() found, using internal fallback. Error:", e)
        ASK_FUNC = None

# ensure directories
LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters")
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

# Create a sample letter if none exists
example_path = os.path.join(LETTERS_DIR, "real.txt")
if not os.path.exists(example_path):
    with open(example_path, "w", encoding="utf-8") as f:
        f.write("HEY!\n\nIf you have any complain kindly mail to 'zultx.service@gmail.com'\n\nIf you like ZultX wanna support it!\n\nWith love,\nAura Sharma\n13\nZultX-Owner.")

# Default UPI (safe placeholder)
UPI_ID = os.getenv("UPI_ID", "9358588509@fam")

app = FastAPI(title="ZULTX — v1.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# serve static if present (optional)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    # prefer index.html in cwd for easy deploy
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<html><body><h1>ZultX</h1><p>UI missing</p></body></html>")


# Local fallback "brain" (synchronous string return)
def local_fallback_ask_plain(user_input: str, mode: str = "friend", temperature: Optional[float] = None,
                             max_tokens: int = 512) -> str:
    base = (user_input or "").strip().lower()
    if any(word in base for word in ("sad", "depressed", "unhappy", "down")):
        answer = (
            "Hey — I'm really sorry you're feeling sad. You're not alone.\n\n"
            "Three quick things that might help right now:\n"
            "1) Take three slow breaths (inhale 4s, hold 2s, exhale 6s).\n"
            "2) Stretch or stand up for 30 seconds.\n"
            "3) Write one small thing you're grateful for.\n\n"
            "If you want, tell me more — I'm here to listen."
        )
    else:
        answer = f"Hey. I heard: \"{user_input}\". Be kind to yourself — tell me more and I'll help."
    return answer


# Utility to consume different result shapes into a single string answer
async def normalize_result_to_text(result: Any) -> str:
    # If result is None -> fallback
    if result is None:
        return local_fallback_ask_plain("")

    # If result is a coroutine (awaitable), await it
    if asyncio.iscoroutine(result):
        try:
            result = await result
        except Exception:
            # if awaiting fails, fallback to string
            return str(result)

    # If result is an async generator
    if hasattr(result, "__aiter__"):
        parts = []
        try:
            async for p in result:
                parts.append(str(p))
        except Exception:
            # best-effort
            pass
        return "".join(parts)

    # If result is a regular generator or iterable (but not string)
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        parts = []
        try:
            for p in result:
                parts.append(str(p))
        except Exception:
            pass
        return "".join(parts)

    # If it's a dict with 'answer' field, prefer it
    if isinstance(result, dict) and "answer" in result:
        return str(result.get("answer") or "")

    # final fallback to str
    return str(result)


@app.get("/ask")
async def ask_get(
    q: str = Query(..., alias="q"),
    mode: str = Query("friend"),
    temperature: Optional[float] = Query(None),
    max_tokens: int = Query(512),
    session_id: Optional[str] = Query(None, description="Session id for conversation buffer"),
    user_id: Optional[str] = Query(None, description="Per-user ID (owner) for memory isolation"),
    memory_mode: str = Query("auto", description="memory mode: auto | manual | off | watch")
):
    """
    /ask?q=...&session_id=xxx&user_id=me&memory_mode=auto
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        if ASK_FUNC is not None:
            try:
                sig = None
                try:
                    import inspect as _inspect
                    sig = _inspect.signature(ASK_FUNC)
                except Exception:
                    sig = None

                kwargs = {
                    "user_input": q,
                    "mode": mode,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    # pass session/user/mode to phase_4
                    "session_id": session_id,
                    "user_id": user_id,
                    "memory_mode": memory_mode
                }
                # only pass 'stream' if function supports it
                if sig and "stream" in sig.parameters:
                    kwargs["stream"] = False

                result = ASK_FUNC(**kwargs)
            except TypeError:
                # fallback: try positional variant
                try:
                    result = ASK_FUNC(q, session_id, user_id, False)
                except Exception:
                    result = ASK_FUNC(q)
        else:
            result = local_fallback_ask_plain(q, mode=mode, temperature=temperature, max_tokens=max_tokens)

        text = await normalize_result_to_text(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"answer": f"ZULTX error: {str(e)}"}, status_code=500)

    if not isinstance(text, str):
        text = str(text)
    if len(text) > 20000:
        text = text[:20000] + "\n\n...[truncated]"

    return JSONResponse({"answer": text})
 

@app.get("/letters")
def list_letters():
    files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(LETTERS_DIR, "*.txt"))])
    return JSONResponse({"letters": files})


@app.get("/letters/{name}")
def get_letter(name: str):
    if ".." in name or not name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(LETTERS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Letter not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)


@app.get("/profile")
def profile():
    return JSONResponse({
        "username": "Guest",
        "display_name": "Guest",
        "email": None,
        "avatar": None,
        "can_logout": True
    })


@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    ts = int(time.time() * 1000)
    fname = f"feedback/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})


@app.post("/tip")
def tip(payload: dict = Body(...)):
    try:
        amount = int(payload.get("amount", 10))
        if amount <= 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount")

    ts = int(time.time())
    tn = urllib.parse.quote_plus("Tip ZULTX")
    upi_uri = f"upi://pay?pa={urllib.parse.quote_plus(UPI_ID)}&pn=ZULTX&tn={tn}&am={amount}&cu=INR"
    qr_payload = urllib.parse.quote_plus(upi_uri)
    qr_url = f"https://chart.googleapis.com/chart?cht=qr&chs=360x360&chl={qr_payload}"

    order = {"id": f"upi_{ts}", "amount": amount * 100, "currency": "INR"}
    try:
        with open(f"tips/{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"order": order, "upi": UPI_ID, "upi_uri": upi_uri, "created_at": ts}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return JSONResponse({"ok": True, "order": order, "upi_link": upi_uri, "qr": qr_url, "note": "Use UPI link or scan QR to pay or Straight to 9358588509@fam."})


@app.post("/tip/confirm")
def tip_confirm(payload: dict = Body(...)):
    ts = int(time.time() * 1000)
    fname = f"tips/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
