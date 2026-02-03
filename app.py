# app.py
import os
import glob
import json
import time
import traceback
import urllib.parse
from typing import Generator, Union, Optional
from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio

# Try to import phase_2.ask -> phase_1.ask -> fallback
ASK_FUNC = None
try:
    from phase_2 import ask as ask_func
    ASK_FUNC = ask_func
    print("[ZULTX] Using phase_2.ask")
except Exception:
    try:
        from phase_1 import ask as ask_func
        ASK_FUNC = ask_func
        print("[ZULTX] Using phase_1.ask")
    except Exception as e:
        print("[ZULTX] No ask() found — falling back to fake stream.", e)
        ASK_FUNC = None

LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters")
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

UPI_ID = os.getenv("UPI_ID", "9358588509@fam")

app = FastAPI(title="ZULTX — v1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<html><body><h1>ZultX</h1><p>UI missing</p></body></html>")


def sse_format(obj: dict) -> str:
    # Format a JSON payload as an SSE "data: {json}\n\n"
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def fake_stream_response(text: str, delay: float = 0.06) -> Generator[str, None, None]:
    """
    Yields text in small chunks wrapped as SSE data events with OpenAI-like structure.
    Each event is: data: {"choices":[{"delta":{"content":"..."} }]}
    """
    # simple split of words/chars to simulate streaming
    i = 0
    n = len(text)
    # Send short initial event (role)
    yield sse_format({"choices":[{"delta":{"role":"assistant","content":""}}]})
    while i < n:
        chunk = text[i: i + 24]  # 24 chars per chunk
        payload = {"choices":[{"delta":{"content":chunk}}]}
        yield sse_format(payload)
        await asyncio.sleep(delay)
        i += 24
    # final stop
    yield sse_format({"choices":[{"delta":{},"finish_reason":"stop"}]})
    yield "data: [DONE]\n\n"


def _to_generator(result: Union[str, Generator[str, None, None]], chunk_size: int = 256):
    """
    If result is iterable (generator), return as-is. If string, return generator that yields the string in chunks.
    """
    if hasattr(result, "__iter__") and not isinstance(result, str):
        return result
    text = str(result or "")
    async def g():
        for i in range(0, len(text), chunk_size):
            yield sse_format({"choices":[{"delta":{"content": text[i:i+chunk_size]}}]})
            await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"
    return g()


@app.get("/ask")
async def ask_get(
    q: str = Query(..., alias="q"),
    mode: str = Query("friend"),
    stream: bool = Query(True),
    speed: float = Query(0.02),
    temperature: Optional[float] = Query(None),
    max_tokens: int = Query(512)
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    # If we have a real ask function and it supports streaming, call it.
    if ASK_FUNC:
        try:
            # Try to call it - it may return string, generator or asyncio generator
            result = ASK_FUNC(user_input=q, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, speed=speed)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": "ZULTX processing failed", "detail": str(e)}, status_code=500)

        # If user requested stream, try to return SSE-like streaming
        if stream:
            # If ASK_FUNC returned a generator of plain text parts, wrap as SSE.
            if hasattr(result, "__aiter__") or hasattr(result, "__iter__"):
                async def stream_gen():
                    # If it's async generator:
                    if hasattr(result, "__aiter__"):
                        async for part in result:
                            # If caller already returns JSON-like parts, try to send safe SSE
                            try:
                                if isinstance(part, dict):
                                    yield sse_format(part)
                                else:
                                    # send as delta content
                                    yield sse_format({"choices":[{"delta":{"content": str(part)}}]})
                            except Exception:
                                yield sse_format({"choices":[{"delta":{"content": str(part)}}]})
                        yield "data: [DONE]\n\n"
                    else:
                        # sync iterator
                        for part in result:
                            yield sse_format({"choices":[{"delta":{"content": str(part)}}]})
                            await asyncio.sleep(speed)
                        yield "data: [DONE]\n\n"
                return StreamingResponse(stream_gen(), media_type="text/event-stream")
            else:
                # If result is a plain string, fake stream it
                return StreamingResponse(fake_stream_response(str(result), delay=speed), media_type="text/event-stream")
        else:
            # non-stream response
            if isinstance(result, str):
                return JSONResponse({"answer": result})
            else:
                # join parts if iterable
                try:
                    text = "".join([p async for p in result]) if hasattr(result, "__aiter__") else "".join(result)
                except Exception:
                    text = str(result)
                return JSONResponse({"answer": text})

    # No ASK_FUNC -> fallback fake-behavior
    fallback_text = f"Echo (offline) — I heard: {q}\n\n(Use phase_2.ask for smarter replies.)"
    if stream:
        return StreamingResponse(fake_stream_response(fallback_text, delay=speed), media_type="text/event-stream")
    else:
        return JSONResponse({"answer": fallback_text})


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

    return JSONResponse({"ok": True, "order": order, "upi_link": upi_uri, "qr": qr_url, "note": "Use UPI link or scan QR to pay."})


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
