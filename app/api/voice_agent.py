"""
Conversational voice agent.

Twilio calls the customer, transcribes their speech (<Gather input="speech">),
and on each turn we ask Claude what the salesperson should say next, speak it
(ElevenLabs, or Twilio's built-in voice), and listen again — until the agent
books a visit, transfers to a human, or the call ends.

Also exposes a few demo helpers:
  POST /demo/call            -> place a live conversational call to a number
  GET  /demo                 -> a tiny browser chat to rehearse the agent (no phone)
  POST /demo/chat            -> one text turn against the same brain
  GET  /demo/call/{id}/transcript -> the full conversation of a call
"""
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from twilio.twiml.voice_response import VoiceResponse, Gather
from datetime import datetime, timezone
from pydantic import BaseModel
import phonenumbers

from app.db.database import get_db, AsyncSessionLocal
from app.db.models import Call, Customer, Campaign, CallStatus, CallOutcome, Appointment
from app.services.voice import generate_and_cache
from app.services.caller import make_call
from app.services import conversation
from app.core.config import settings

router = APIRouter(prefix="/voice/agent", tags=["voice-agent"])
demo_router = APIRouter(prefix="/demo", tags=["demo"])


# ----------------------------- speech helpers -----------------------------

async def _say(node, text: str, cache_key: str):
    """Append spoken text to a VoiceResponse or Gather, via ElevenLabs or Twilio."""
    use_eleven = (
        settings.TTS_PROVIDER == "elevenlabs"
        and settings.ELEVENLABS_API_KEY
        and settings.ELEVENLABS_VOICE_ID
    )
    if use_eleven:
        try:
            url = await generate_and_cache(text, cache_key)
            node.play(url)
            return
        except Exception as e:  # fall back so a TTS hiccup never kills the call
            print(f"[tts] ElevenLabs failed ({e}); using Twilio voice")
    node.say(text, voice=settings.TWILIO_VOICE)


def _first_name(customer: Customer | None) -> str:
    if customer and customer.name:
        return customer.name.split()[0]
    return ""


async def _gather_response(text: str, call_id: int, turn: int, attempts: int = 0) -> str:
    """Speak `text`, then listen for the customer's reply."""
    vr = VoiceResponse()
    gather = Gather(
        input="speech",
        speech_timeout="auto",
        action=f"{settings.APP_BASE_URL}/voice/agent/turn?call_id={call_id}&attempts={attempts}",
        method="POST",
        action_on_empty_result=True,
    )
    await _say(gather, text, f"c{call_id}_t{turn}")
    vr.append(gather)
    return str(vr)


async def _say_and_hangup(text: str, call_id: int, turn: int) -> str:
    vr = VoiceResponse()
    await _say(vr, text, f"c{call_id}_t{turn}")
    vr.hangup()
    return str(vr)


def _xml(twiml: str) -> PlainTextResponse:
    return PlainTextResponse(twiml, media_type="application/xml")


async def _load_call(call_id: int, db: AsyncSession):
    result = await db.execute(
        select(Call, Customer).join(Customer, Call.customer_id == Customer.id).where(Call.id == call_id)
    )
    row = result.first()
    return (row[0], row[1]) if row else (None, None)


def _print_turn(call_id: int, speaker: str, text: str):
    print(f"\n📞 [call {call_id}] {speaker}: {text}", flush=True)


# ----------------------------- Twilio webhooks -----------------------------

@router.post("/start")
async def agent_start(request: Request, call_id: int, db: AsyncSession = Depends(get_db)):
    """Call connected — the agent greets the customer and starts the conversation."""
    form = await request.form()
    answered_by = form.get("AnsweredBy", "human")

    call, customer = await _load_call(call_id, db)
    if not call:
        return _xml("<Response><Hangup/></Response>")

    first = _first_name(customer)

    # Voicemail detected -> leave a short message and hang up.
    if answered_by in ("machine_start", "machine_end_beep", "machine_end_silence"):
        vm = (
            f"Hi {first or 'there'}, this is {settings.AGENT_NAME} calling on behalf of "
            f"{settings.BUSINESS_NAME}. We're running {settings.SALE_HEADLINE} and we'd love to "
            f"set you up with a free in-home measure. Give us a call back if you're interested. Thanks!"
        )
        call.status = CallStatus.voicemail
        call.outcome = CallOutcome.voicemail_left
        call.started_at = datetime.now(timezone.utc)
        await db.commit()
        _print_turn(call_id, "voicemail", vm)
        return _xml(await _say_and_hangup(vm, call_id, 0))

    # Human answered — generate the opening line from Claude.
    history = conversation.initial_history(first, customer_phone=customer.phone if customer else "")
    result = await conversation.next_reply(history, first, customer_phone=customer.phone if customer else "")
    history.append({"role": "assistant", "content": result["reply"]})

    call.status = CallStatus.in_progress
    call.started_at = datetime.now(timezone.utc)
    call.transcript = history
    await db.commit()

    _print_turn(call_id, settings.AGENT_NAME, result["reply"])
    return _xml(await _gather_response(result["reply"], call_id, turn=len(history)))


@router.post("/turn")
async def agent_turn(request: Request, call_id: int, attempts: int = 0, db: AsyncSession = Depends(get_db)):
    """Customer spoke — figure out the agent's next line and speak it."""
    form = await request.form()
    speech = (form.get("SpeechResult") or "").strip()

    call, customer = await _load_call(call_id, db)
    if not call:
        return _xml("<Response><Hangup/></Response>")

    first = _first_name(customer)
    history = list(call.transcript or [])

    # No speech captured — re-prompt once or twice, then wrap up.
    if not speech:
        if attempts >= 2:
            bye = "I'll let you go for now. Thanks so much, and have a wonderful day!"
            call.status = CallStatus.completed
            await db.commit()
            _print_turn(call_id, settings.AGENT_NAME, bye)
            return _xml(await _say_and_hangup(bye, call_id, turn=len(history) + 1))
        nudge = "Sorry, I didn't quite catch that — are you still there?"
        _print_turn(call_id, settings.AGENT_NAME, f"(no speech) {nudge}")
        return _xml(await _gather_response(nudge, call_id, turn=len(history) + 1, attempts=attempts + 1))

    _print_turn(call_id, customer.name or "Customer", speech)
    history.append({"role": "user", "content": speech})

    result = await conversation.next_reply(history, first, customer_phone=customer.phone if customer else "")
    reply, action, detail = result["reply"], result["action"], result["detail"]
    history.append({"role": "assistant", "content": reply})
    call.transcript = history
    _print_turn(call_id, settings.AGENT_NAME, reply + (f"  ⟶ [{action} {detail}]" if action else ""))

    if action == "BOOK":
        db.add(Appointment(
            customer_id=call.customer_id,
            call_id=call.id,
            scheduled_at=datetime.now(timezone.utc),
            notes=f"Free in-home measure requested: {detail}" if detail else "Measure booked during call",
        ))
        call.outcome = CallOutcome.appointment_booked
        call.status = CallStatus.completed
        await db.commit()
        return _xml(await _say_and_hangup(reply, call_id, turn=len(history)))

    if action == "DO_NOT_CALL":
        # Honor the request immediately and permanently.
        customer.do_not_call = True
        call.outcome = CallOutcome.not_interested
        call.status = CallStatus.do_not_call
        call.notes = "Customer asked to be removed (do-not-call)."
        await db.commit()
        return _xml(await _say_and_hangup(reply, call_id, turn=len(history)))

    if action in ("INTERESTED", "CALLBACK", "NOT_INTERESTED", "WRONG_NUMBER", "BAD_TIMING"):
        call.outcome = {
            "INTERESTED":    CallOutcome.interested,
            "CALLBACK":      CallOutcome.callback_requested,
            "NOT_INTERESTED": CallOutcome.not_interested,
            "WRONG_NUMBER":  CallOutcome.wrong_number,
            "BAD_TIMING":    CallOutcome.bad_timing,
        }[action]
        call.status = CallStatus.completed
        if detail:
            call.notes = f"{action.lower()}: {detail}"
        await db.commit()
        return _xml(await _say_and_hangup(reply, call_id, turn=len(history)))

    if action == "TRANSFER":
        call.outcome = CallOutcome.transferred
        call.status = CallStatus.completed
        await db.commit()
        vr = VoiceResponse()
        await _say(vr, reply, f"c{call_id}_t{len(history)}")
        if settings.STORE_PHONE:
            vr.dial(settings.STORE_PHONE)
        else:
            vr.hangup()
        return _xml(str(vr))

    if action == "END":
        call.status = CallStatus.completed
        await db.commit()
        return _xml(await _say_and_hangup(reply, call_id, turn=len(history)))

    # Normal turn — keep the conversation going.
    await db.commit()
    return _xml(await _gather_response(reply, call_id, turn=len(history)))


@router.post("/status")
async def agent_status(request: Request, call_id: int, db: AsyncSession = Depends(get_db)):
    """Twilio status callback — record final status and duration."""
    form = await request.form()
    twilio_status = form.get("CallStatus", "")
    duration = form.get("CallDuration", 0)

    call, _ = await _load_call(call_id, db)
    if not call:
        return PlainTextResponse("OK")

    status_map = {
        "no-answer": CallStatus.no_answer,
        "busy": CallStatus.no_answer,
        "failed": CallStatus.failed,
        "canceled": CallStatus.failed,
    }
    if twilio_status in status_map:
        call.status = status_map[twilio_status]
    elif call.status == CallStatus.in_progress:
        call.status = CallStatus.completed

    try:
        call.duration_seconds = int(duration)
    except (TypeError, ValueError):
        pass
    call.ended_at = datetime.now(timezone.utc)
    await db.commit()
    return PlainTextResponse("OK")


# ----------------------------- demo helpers -----------------------------

class DemoCall(BaseModel):
    phone: str
    name: str = "there"


def _to_e164(raw: str) -> str:
    try:
        parsed = phonenumbers.parse(raw, "US")
    except phonenumbers.NumberParseException:
        raise HTTPException(400, f"'{raw}' is not a valid phone number")
    if not phonenumbers.is_valid_number(parsed):
        raise HTTPException(400, f"'{raw}' is not a valid phone number")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


async def _get_demo_campaign(db: AsyncSession) -> Campaign:
    result = await db.execute(select(Campaign).where(Campaign.name == "Voice Demo"))
    campaign = result.scalar_one_or_none()
    if not campaign:
        campaign = Campaign(name="Voice Demo", script_key="conversation")
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)
    return campaign


@demo_router.post("/call")
async def demo_call(data: DemoCall, db: AsyncSession = Depends(get_db)):
    """Place a live conversational call to a phone number. Returns the call id."""
    e164 = _to_e164(data.phone)
    campaign = await _get_demo_campaign(db)

    result = await db.execute(select(Customer).where(Customer.phone == e164))
    customer = result.scalar_one_or_none()
    if customer:
        # Honor the do-not-call list — never re-dial someone who opted out.
        if customer.do_not_call:
            raise HTTPException(403, f"{e164} is on the do-not-call list and will not be dialed.")
        customer.name = data.name
    else:
        customer = Customer(name=data.name, phone=e164)
        db.add(customer)
    await db.commit()
    await db.refresh(customer)

    call = Call(
        customer_id=customer.id,
        campaign_id=campaign.id,
        status=CallStatus.pending,
        scheduled_at=datetime.now(timezone.utc),
        transcript=[],
    )
    db.add(call)
    await db.commit()
    await db.refresh(call)

    try:
        sid = make_call(e164, call.id, intro_path="/voice/agent/start")
        call.twilio_call_sid = sid
        call.status = CallStatus.in_progress
        await db.commit()
    except Exception as e:
        call.status = CallStatus.failed
        call.notes = str(e)
        await db.commit()
        raise HTTPException(500, f"Could not place call: {e}")

    return {
        "call_id": call.id,
        "calling": e164,
        "transcript_url": f"{settings.APP_BASE_URL}/demo/call/{call.id}/transcript",
        "message": f"Calling {data.name} now — watch your server console for the live conversation.",
    }


@demo_router.get("/call/{call_id}/transcript")
async def call_transcript(call_id: int, db: AsyncSession = Depends(get_db)):
    """The full conversation for a call (agent + customer turns)."""
    call, customer = await _load_call(call_id, db)
    if not call:
        raise HTTPException(404, "Call not found")
    turns = []
    for t in (call.transcript or []):
        content = t.get("content", "")
        if isinstance(content, str) and content.startswith("<call_connected>"):
            continue  # hide the internal kickoff prompt
        turns.append({
            "speaker": settings.AGENT_NAME if t.get("role") == "assistant" else (customer.name or "Customer"),
            "text": content,
        })
    return {
        "call_id": call.id,
        "customer": customer.name if customer else None,
        "status": call.status.value if call.status else None,
        "outcome": call.outcome.value if call.outcome else None,
        "turns": turns,
    }


# --- text simulator: rehearse the agent in a browser, no phone needed ---

_sim_sessions: dict[str, dict] = {}


class ChatTurn(BaseModel):
    session_id: str = "default"
    message: str = ""
    name: str = "there"


@demo_router.post("/chat")
async def demo_chat(turn: ChatTurn):
    """One text turn against the live agent brain. Empty message starts the call."""
    sess = _sim_sessions.get(turn.session_id)
    if sess is None or not turn.message:
        history = conversation.initial_history(turn.name)
        result = await conversation.next_reply(history, turn.name)
        history.append({"role": "assistant", "content": result["reply"]})
        _sim_sessions[turn.session_id] = {"history": history, "name": turn.name}
        return {"reply": result["reply"], "action": result["action"], "detail": result["detail"]}

    history = sess["history"]
    history.append({"role": "user", "content": turn.message})
    result = await conversation.next_reply(history, sess["name"])
    history.append({"role": "assistant", "content": result["reply"]})
    if result["action"] in (
        "BOOK", "INTERESTED", "CALLBACK", "NOT_INTERESTED",
        "DO_NOT_CALL", "WRONG_NUMBER", "BAD_TIMING", "TRANSFER", "END"
    ):
        _sim_sessions.pop(turn.session_id, None)  # conversation finished
    return {"reply": result["reply"], "action": result["action"], "detail": result["detail"]}


@demo_router.get("", response_class=HTMLResponse)
async def demo_page():
    html = (
        _CHAT_HTML
        .replace("__AGENT__", settings.AGENT_NAME)
        .replace("__BUSINESS__", settings.BUSINESS_NAME)
    )
    return HTMLResponse(html)


_CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__BUSINESS__ — AI Voice Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --blue:       #2563EB;
    --blue-light: #EFF6FF;
    --blue-dark:  #1D4ED8;
    --green:      #059669;
    --green-light:#ECFDF5;
    --red:        #DC2626;
    --red-light:  #FEF2F2;
    --amber:      #D97706;
    --amber-light:#FFFBEB;
    --slate-900:  #0F172A;
    --slate-700:  #334155;
    --slate-500:  #64748B;
    --slate-300:  #CBD5E1;
    --slate-100:  #F1F5F9;
    --white:      #FFFFFF;
    --bg:         #F8FAFF;
  }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--slate-900);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── NAV ─────────────────────────────────────────────── */
  nav {
    background: var(--white);
    border-bottom: 1px solid var(--slate-100);
    padding: 0 32px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 10;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }
  .nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 16px;
    color: var(--slate-900);
    text-decoration: none;
  }
  .nav-logo {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 15px;
  }
  .nav-badge {
    background: var(--blue-light);
    color: var(--blue);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: .3px;
  }

  /* ── HERO ─────────────────────────────────────────────── */
  .hero {
    text-align: center;
    padding: 56px 24px 40px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(37,99,235,.08) 0%, transparent 70%);
    top: -200px; left: 50%; transform: translateX(-50%);
    pointer-events: none;
  }
  .hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--blue-light);
    color: var(--blue);
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 20px;
    letter-spacing: .4px;
    text-transform: uppercase;
  }
  .hero-eyebrow::before {
    content: '';
    width: 7px; height: 7px;
    background: var(--blue);
    border-radius: 50%;
    animation: pulse 1.8s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(.7); }
  }
  .hero h1 {
    font-size: clamp(26px, 4vw, 40px);
    font-weight: 700;
    line-height: 1.15;
    color: var(--slate-900);
    margin-bottom: 14px;
  }
  .hero h1 span { color: var(--blue); }
  .hero p {
    font-size: 15px;
    color: var(--slate-500);
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* ── MAIN LAYOUT ─────────────────────────────────────── */
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    max-width: 1040px;
    margin: 0 auto;
    padding: 0 24px 48px;
    width: 100%;
    flex: 1;
  }

  /* ── SIDEBAR ─────────────────────────────────────────── */
  .sidebar { display: flex; flex-direction: column; gap: 16px; }

  .card {
    background: var(--white);
    border: 1px solid var(--slate-100);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
  }

  .agent-card { text-align: center; padding: 24px 20px; }
  .agent-avatar {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 24px;
    margin: 0 auto 12px;
    position: relative;
  }
  .agent-avatar::after {
    content: '';
    position: absolute;
    bottom: 2px; right: 2px;
    width: 14px; height: 14px;
    background: var(--green);
    border: 2px solid white;
    border-radius: 50%;
  }
  .agent-name { font-size: 17px; font-weight: 700; margin-bottom: 3px; }
  .agent-role { font-size: 12px; color: var(--slate-500); margin-bottom: 14px; }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--green-light);
    color: var(--green);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
  }
  .status-dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
  }

  .card-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: var(--slate-500);
    margin-bottom: 12px;
  }
  .deal-box {
    background: linear-gradient(135deg, #EFF6FF, #F5F3FF);
    border: 1px solid #DBEAFE;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    margin-bottom: 10px;
  }
  .deal-pct {
    font-size: 36px;
    font-weight: 800;
    color: var(--blue);
    line-height: 1;
    margin-bottom: 2px;
  }
  .deal-desc {
    font-size: 11px;
    font-weight: 600;
    color: var(--slate-700);
    text-transform: uppercase;
    letter-spacing: .5px;
  }
  .deal-sub {
    font-size: 11px;
    color: var(--slate-500);
    margin-top: 6px;
  }

  .rule-list { list-style: none; display: flex; flex-direction: column; gap: 8px; }
  .rule-list li {
    display: flex; align-items: flex-start; gap: 8px;
    font-size: 12px; color: var(--slate-700); line-height: 1.4;
  }
  .rule-list .dot {
    width: 18px; height: 18px; min-width: 18px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    margin-top: 1px;
  }
  .dot-green { background: var(--green-light); color: var(--green); }
  .dot-red   { background: var(--red-light);   color: var(--red);   }

  /* ── CHAT PANEL ──────────────────────────────────────── */
  .chat-panel {
    background: var(--white);
    border: 1px solid var(--slate-100);
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,.06);
    min-height: 560px;
  }
  .chat-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--slate-100);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--white);
  }
  .chat-header-left {
    display: flex; align-items: center; gap: 10px;
  }
  .chat-header-avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px;
  }
  .chat-header-name { font-size: 14px; font-weight: 600; }
  .chat-header-sub  { font-size: 11px; color: var(--slate-500); }
  .chat-header-badge {
    font-size: 11px; font-weight: 600;
    color: var(--slate-500);
    background: var(--slate-100);
    padding: 4px 10px;
    border-radius: 20px;
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }
  .chat-messages::-webkit-scrollbar { width: 4px; }
  .chat-messages::-webkit-scrollbar-track { background: transparent; }
  .chat-messages::-webkit-scrollbar-thumb { background: var(--slate-300); border-radius: 4px; }

  .msg { display: flex; gap: 10px; max-width: 82%; }
  .msg.agent { align-self: flex-start; }
  .msg.user  { align-self: flex-end; flex-direction: row-reverse; }

  .msg-avatar {
    width: 30px; height: 30px; min-width: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    margin-top: 2px;
  }
  .msg.agent .msg-avatar { background: linear-gradient(135deg,#2563EB,#7C3AED); color:white; }
  .msg.user  .msg-avatar { background: var(--slate-100); color: var(--slate-500); }

  .msg-body { display: flex; flex-direction: column; gap: 4px; }

  .msg-name { font-size: 11px; font-weight: 600; color: var(--slate-500); }
  .msg.user .msg-name { text-align: right; }

  .bubble {
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.55;
    word-wrap: break-word;
  }
  .msg.agent .bubble {
    background: var(--blue-light);
    color: var(--slate-900);
    border-bottom-left-radius: 4px;
  }
  .msg.user .bubble {
    background: var(--blue);
    color: white;
    border-bottom-right-radius: 4px;
  }

  .outcome-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: .4px;
    width: fit-content;
  }
  .outcome-BOOK, .outcome-INTERESTED { background: var(--green-light); color: var(--green); }
  .outcome-DO_NOT_CALL, .outcome-WRONG_NUMBER { background: var(--red-light); color: var(--red); }
  .outcome-CALLBACK, .outcome-BAD_TIMING { background: var(--amber-light); color: var(--amber); }
  .outcome-NOT_INTERESTED, .outcome-END, .outcome-TRANSFER { background: var(--slate-100); color: var(--slate-500); }

  .typing-indicator {
    display: flex; gap: 4px; align-items: center;
    padding: 14px 18px;
    background: var(--blue-light);
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    width: fit-content;
  }
  .typing-dot {
    width: 6px; height: 6px;
    background: var(--blue);
    border-radius: 50%;
    animation: typing 1.2s infinite;
  }
  .typing-dot:nth-child(2) { animation-delay: .2s; }
  .typing-dot:nth-child(3) { animation-delay: .4s; }
  @keyframes typing {
    0%, 60%, 100% { transform: translateY(0); opacity: .4; }
    30%            { transform: translateY(-5px); opacity: 1; }
  }

  .chat-input-area {
    padding: 16px 20px;
    border-top: 1px solid var(--slate-100);
    display: flex;
    gap: 10px;
    align-items: center;
    background: var(--white);
  }
  .chat-input-area input {
    flex: 1;
    padding: 12px 18px;
    border: 1.5px solid var(--slate-300);
    border-radius: 100px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: var(--slate-900);
    outline: none;
    transition: border-color .15s;
    background: var(--bg);
  }
  .chat-input-area input:focus { border-color: var(--blue); background: var(--white); }
  .chat-input-area input::placeholder { color: var(--slate-500); }
  .send-btn {
    width: 42px; height: 42px; min-width: 42px;
    background: var(--blue);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .15s, transform .1s;
    color: white;
  }
  .send-btn:hover { background: var(--blue-dark); }
  .send-btn:active { transform: scale(.93); }
  .send-btn svg { width: 18px; height: 18px; fill: none; stroke: white; stroke-width: 2; }

  .hint-row {
    display: flex; gap: 8px; flex-wrap: wrap;
    padding: 0 20px 14px;
  }
  .hint {
    font-size: 11px;
    color: var(--blue);
    background: var(--blue-light);
    padding: 4px 10px;
    border-radius: 20px;
    cursor: pointer;
    border: none;
    font-family: inherit;
    transition: background .15s;
    white-space: nowrap;
  }
  .hint:hover { background: #DBEAFE; }

  /* ── RESPONSIVE ──────────────────────────────────────── */
  @media (max-width: 700px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { flex-direction: row; flex-wrap: wrap; }
    .sidebar .card { flex: 1 1 220px; }
    .hero { padding: 36px 20px 28px; }
    nav { padding: 0 16px; }
  }

  /* ── CALL ENDED STATE ────────────────────────────────── */
  .ended-banner {
    display: none;
    margin: 16px 20px 0;
    padding: 12px 16px;
    background: var(--slate-100);
    border-radius: 12px;
    font-size: 13px;
    color: var(--slate-700);
    text-align: center;
  }
  .restart-btn {
    background: none;
    border: 1.5px solid var(--blue);
    color: var(--blue);
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 20px;
    border-radius: 100px;
    cursor: pointer;
    margin-top: 8px;
    transition: background .15s;
    display: block;
    margin: 8px auto 0;
  }
  .restart-btn:hover { background: var(--blue-light); }
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <a class="nav-brand" href="#">
    <div class="nav-logo">🎙</div>
    <span>__BUSINESS__</span>
  </a>
  <span class="nav-badge">Live Call Simulation</span>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-eyebrow">AI Voice Agent</div>
  <h1>Talk to <span>__AGENT__</span>, our AI Sales Agent</h1>
  <p>Type as the customer. The same AI brain that runs on real outbound calls will respond — in real time.</p>
</div>

<!-- MAIN -->
<div class="layout">

  <!-- SIDEBAR -->
  <div class="sidebar">

    <div class="card agent-card">
      <div class="agent-avatar">🎙</div>
      <div class="agent-name">__AGENT__</div>
      <div class="agent-role">AI Sales Agent · __BUSINESS__</div>
      <div class="status-pill"><span class="status-dot"></span>Online</div>
    </div>

    <div class="card">
      <div class="card-label">Weekend Offer</div>
      <div class="deal-box">
        <div class="deal-pct">40%</div>
        <div class="deal-desc">Off All Carpets</div>
        <div class="deal-sub">Free in-home measure included</div>
      </div>
      <div style="font-size:11px;color:var(--slate-500);text-align:center;">This weekend only · No obligations</div>
    </div>

    <div class="card">
      <div class="card-label">Agent Guardrails</div>
      <ul class="rule-list">
        <li><span class="dot dot-green">✓</span>Stays on facts — exactly 40%, no extras</li>
        <li><span class="dot dot-green">✓</span>Discloses AI when asked</li>
        <li><span class="dot dot-green">✓</span>Honors do-not-call immediately</li>
        <li><span class="dot dot-green">✓</span>Books via real calendar availability</li>
        <li><span class="dot dot-red">✗</span>Never invents prices or terms</li>
        <li><span class="dot dot-red">✗</span>Never pressures the customer</li>
      </ul>
    </div>

  </div>

  <!-- CHAT -->
  <div class="chat-panel">
    <div class="chat-header">
      <div class="chat-header-left">
        <div class="chat-header-avatar">🎙</div>
        <div>
          <div class="chat-header-name">__AGENT__ — __BUSINESS__</div>
          <div class="chat-header-sub">Outbound call simulation</div>
        </div>
      </div>
      <span class="chat-header-badge">Demo Mode</span>
    </div>

    <div class="chat-messages" id="messages"></div>

    <div class="ended-banner" id="ended-banner">
      Call ended.
      <button class="restart-btn" onclick="restart()">Start a new call</button>
    </div>

    <div class="hint-row" id="hints">
      <button class="hint" onclick="useHint(this)">What's included?</button>
      <button class="hint" onclick="useHint(this)">How much will it cost?</button>
      <button class="hint" onclick="useHint(this)">I'm not interested.</button>
      <button class="hint" onclick="useHint(this)">Saturday morning works.</button>
      <button class="hint" onclick="useHint(this)">Remove me from your list.</button>
    </div>

    <div class="chat-input-area">
      <input id="input" placeholder="Type what the customer says…" autocomplete="off">
      <button class="send-btn" onclick="sendMsg()" aria-label="Send">
        <svg viewBox="0 0 24 24"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
    </div>
  </div>

</div>

<script>
const SID = 'web-' + Math.random().toString(36).slice(2);
const AGENT = '__AGENT__';
let ended = false;

const $msgs    = document.getElementById('messages');
const $input   = document.getElementById('input');
const $banner  = document.getElementById('ended-banner');
const $hints   = document.getElementById('hints');

const OUTCOME_LABELS = {
  BOOK:'Appointment Booked', INTERESTED:'Interested', CALLBACK:'Callback Requested',
  NOT_INTERESTED:'Not Interested', DO_NOT_CALL:'Do Not Call', WRONG_NUMBER:'Wrong Number',
  BAD_TIMING:'Bad Timing', TRANSFER:'Transferred', END:'Call Ended'
};

function addMsg(who, text, isAgent, action) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + (isAgent ? 'agent' : 'user');

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.textContent = isAgent ? '🎙' : '👤';

  const body = document.createElement('div');
  body.className = 'msg-body';

  const name = document.createElement('div');
  name.className = 'msg-name';
  name.textContent = isAgent ? AGENT : 'You';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  body.appendChild(name);
  body.appendChild(bubble);

  if (action && OUTCOME_LABELS[action]) {
    const badge = document.createElement('div');
    badge.className = 'outcome-badge outcome-' + action;
    badge.textContent = OUTCOME_LABELS[action];
    body.appendChild(badge);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(body);
  $msgs.appendChild(wrap);
  $msgs.scrollTop = $msgs.scrollHeight;
}

function showTyping() {
  const t = document.createElement('div');
  t.id = 'typing';
  t.className = 'msg agent';
  t.innerHTML = \`<div class="msg-avatar">🎙</div>
    <div class="msg-body">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>\`;
  $msgs.appendChild(t);
  $msgs.scrollTop = $msgs.scrollHeight;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

async function callAPI(msg) {
  const res = await fetch('/demo/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: SID, message: msg, name: '' })
  });
  return res.json();
}

async function sendMsg() {
  if (ended) return;
  const txt = $input.value.trim();
  if (!txt) return;
  addMsg('You', txt, false);
  $input.value = '';
  $hints.style.display = 'none';
  showTyping();
  const d = await callAPI(txt);
  removeTyping();
  addMsg(AGENT, d.reply, true, d.action);
  const terminal = ['BOOK','INTERESTED','CALLBACK','NOT_INTERESTED','DO_NOT_CALL','WRONG_NUMBER','BAD_TIMING','TRANSFER','END'];
  if (terminal.includes(d.action)) { ended = true; $banner.style.display = 'block'; $input.disabled = true; }
}

function useHint(el) {
  $input.value = el.textContent;
  $input.focus();
}

function restart() {
  ended = false;
  $msgs.innerHTML = '';
  $banner.style.display = 'none';
  $hints.style.display = 'flex';
  $input.disabled = false;
  init();
}

async function init() {
  showTyping();
  const d = await callAPI('');
  removeTyping();
  addMsg(AGENT, d.reply, true);
}

$input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMsg(); });
init();
</script>
</body>
</html>"""
