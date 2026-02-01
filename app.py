# app.py
"""
ZULTX v1.1 backend — UPI (FamPay) tip flow
- streaming /ask using phase_2.ask (fallback phase_1.ask)
- letters, profile, feedback
- /tip -> returns upi link + qr + order_id
- /tip/confirm -> record successful payments (client-posted)
"""
import os
import glob
import json
import time
import traceback
from typing import Generator, Union, Optional
import urllib.parse

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Try to import phase_2 (adapters), else phase_1
ASK_FUNC = None
try:
    from phase_2 import ask as ask_func
    ASK_FUNC = ask_func
    print("[ZULTX] Using phase_2.ask")
except Exception:
    try:
        from phase_1 import ask as ask_func
        ASK_FUNC = ask_func
        print("[ZULTX] Using phase_1.ask (phase_2 not found)")
    except Exception as e:
        print("[ZULTX WARNING] No ask() available:", e)
        ASK_FUNC = None

# Letters folder & feedback/tips storage
LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters")
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

# UPI config (set in Railway env if you want)
UPI_ID = os.getenv("UPI_ID", "9358588509@fam")  # default to your FamPay ID

app = FastAPI(title="ZULTX — v1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


def _to_generator(result: Union[str, Generator[str, None, None]], chunk_size: int = 256):
    if hasattr(result, "__iter__") and not isinstance(result, str):
        return result
    text = str(result or "")
    def g():
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]
            time.sleep(0.01)
    return g()


@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<html><body><h1>ZultX</h1><p>UI missing</p></body></html>")


@app.get("/ask")
def ask_get(
    q: str = Query(..., alias="q"),
    mode: str = Query("friend"),
    stream: bool = Query(True),
    speed: float = Query(0.02),
    temperature: Optional[float] = Query(None),
    max_tokens: int = Query(512)
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")
    if ASK_FUNC is None:
        return JSONResponse({"error": "ZULTX brain not available on server"}, status_code=500)

    try:
        result = ASK_FUNC(
            user_input=q,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            speed=speed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": "ZULTX processing failed", "detail": str(e)}, status_code=500)

    if stream:
        gen = _to_generator(result)
        return StreamingResponse(gen, media_type="text/plain")
    else:
        if isinstance(result, str):
            return JSONResponse({"answer": result})
        else:
            text = "".join([chunk for chunk in result])
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
    """
    Create a tip order for UPI:
    Returns: { ok, order_id, upi_link, qr }
    Client: redirect mobile to upi_link (upi://...) OR show qr (qr url)
    """
    try:
        amount = int(payload.get("amount", 10))
        if amount <= 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount")

    ts = int(time.time())
    # Build UPI URI (upi://pay) — compatible with most UPI apps
    # pa = payee address, pn = payee name, tn = transaction note, am = amount, cu = currency
    tn = urllib.parse.quote_plus("Tip ZULTX")
    upi_uri = f"upi://pay?pa={urllib.parse.quote_plus(UPI_ID)}&pn=ZULTX&tn={tn}&am={amount}&cu=INR"
    # Build QR image via Google Chart API (safe client-side option)
    qr_payload = urllib.parse.quote_plus(upi_uri)
    qr_url = f"https://chart.googleapis.com/chart?cht=qr&chs=360x360&chl={qr_payload}"

    order = {"id": f"upi_{ts}", "amount": amount * 100, "currency": "INR"}  # amount in paise-ish for records
    # Save a basic order record (not money-sensitive)
    try:
        with open(f"tips/{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"order": order, "upi": UPI_ID, "upi_uri": upi_uri, "created_at": ts}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return JSONResponse({"ok": True, "order": order, "upi_link": upi_uri, "qr": qr_url, "note": "Use UPI link or scan QR to pay."})


@app.post("/tip/confirm")
def tip_confirm(payload: dict = Body(...)):
    """
    Record a confirmed payment from client (post-check).
    payload should include: order_id, payment_id (optional), amount
    """
    ts = int(time.time() * 1000)
    fname = f"tips/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
