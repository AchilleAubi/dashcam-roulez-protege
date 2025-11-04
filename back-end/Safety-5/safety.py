from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import json, time, datetime, uuid

ROOT = Path(__file__).parent
DATA = ROOT / "data"
EVIDENCE = ROOT / "evidence"
DATA.mkdir(exist_ok=True, parents=True)
EVIDENCE.mkdir(exist_ok=True, parents=True)

CONTACTS_JSON  = DATA / "contacts.json"
INCIDENTS_JSON = DATA / "incidents.json"
PRIVACY_JSON   = DATA / "privacy.json"

def _load(p: Path, default):
    if not p.exists(): return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# init files if empty
if not CONTACTS_JSON.exists():  _save(CONTACTS_JSON, [])
if not INCIDENTS_JSON.exists(): _save(INCIDENTS_JSON, [])
if not PRIVACY_JSON.exists():   _save(PRIVACY_JSON, {"enabled": False, "locked_at": None})

app = FastAPI(title="Safety-5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/evidence", StaticFiles(directory=str(EVIDENCE)), name="evidence")


class ContactIn(BaseModel):
    name: str
    phone: str | None = None
    email: str | None = None
    channels: list[str] = Field(default_factory=lambda: ["email"])

class Location(BaseModel):
    lat: float | None = None
    lng: float | None = None
    address: str | None = None

class SosIn(BaseModel):
    location: Location | None = None
    note: str | None = None
    attachRecordingName: str | None = None

class CrashIn(BaseModel):
    gForce: float
    speedKmh: float | None = None
    location: Location | None = None

class PrivacyIn(BaseModel):
    enabled: bool


def _new_id():
    return uuid.uuid4().hex[:10]

def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _notify(incident, contacts):
    # MVP: simulation (print + journal de l’incident)
    for c in contacts:
        print(f"[NOTIFY] -> {c['name']} via {','.join(c['channels'])}: Incident #{incident['id']} {incident['type']} severity={incident['severity']}")
    incident["contacts_notified"] = [c["id"] for c in contacts]

def _attach_recording(incident, name):
    url = f"http://localhost:8000/files/{name}"
    incident.setdefault("media", []).append(url)

def _create_incident(itype: str, severity: int, location: dict | None, note: str | None = None):
    incidents = _load(INCIDENTS_JSON, [])
    payload = {
        "id": _new_id(),
        "timestamp": _now_iso(),
        "type": itype,
        "severity": severity,
        "location": location or {},
        "note": note,
        "status": "open",
        "media": [],
        "contacts_notified": [],
    }
    incidents.append(payload)
    _save(INCIDENTS_JSON, incidents)
    return payload


@app.get("/api/emergency/contacts")
def list_contacts():
    return _load(CONTACTS_JSON, [])

@app.post("/api/emergency/contacts")
def add_contact(c: ContactIn):
    contacts = _load(CONTACTS_JSON, [])
    row = dict(id=_new_id(), **c.dict())
    contacts.append(row)
    _save(CONTACTS_JSON, contacts)
    return row

@app.delete("/api/emergency/contacts/{cid}")
def del_contact(cid: str):
    contacts = _load(CONTACTS_JSON, [])
    newc = [c for c in contacts if c["id"] != cid]
    if len(newc) == len(contacts):
        raise HTTPException(status_code=404, detail="Contact introuvable")
    _save(CONTACTS_JSON, newc)
    return {"ok": True}


@app.post("/api/emergency/sos")
def sos(body: SosIn):
    inc = _create_incident("sos", severity=70, location=body.location.dict() if body.location else None, note=body.note)
    if body.attachRecordingName:
        _attach_recording(inc, body.attachRecordingName)

    contacts = _load(CONTACTS_JSON, [])
    _notify(inc, contacts)
    inc["status"] = "sent"
    incidents = _load(INCIDENTS_JSON, [])
    for i in incidents:
        if i["id"] == inc["id"]:
            i.update(inc)
    _save(INCIDENTS_JSON, incidents)
    return inc

@app.post("/api/emergency/crash-event")
def crash_event(body: CrashIn):
    # règle simple de sévérité : min(100, gForce * 30 + speed/2)
    speed = body.speedKmh or 0
    severity = int(min(100, body.gForce * 30 + speed / 2))
    auto_sos = body.gForce >= 2.5

    inc = _create_incident("crash", severity=severity, location=body.location.dict() if body.location else None)
    if auto_sos:
        contacts = _load(CONTACTS_JSON, [])
        _notify(inc, contacts)
        inc["status"] = "sent"
        # persist update
        incidents = _load(INCIDENTS_JSON, [])
        for i in incidents:
            if i["id"] == inc["id"]:
                i.update(inc)
        _save(INCIDENTS_JSON, incidents)
    return {"incident": inc, "autoSOS": auto_sos}

@app.get("/api/emergency/incidents")
def list_incidents():
    # tri du + récent au + ancien
    data = _load(INCIDENTS_JSON, [])
    data.sort(key=lambda x: x.get("timestamp",""), reverse=True)
    return data

@app.get("/api/emergency/incidents/{iid}")
def get_incident(iid: str):
    data = _load(INCIDENTS_JSON, [])
    for it in data:
        if it["id"] == iid:
            return it
    raise HTTPException(status_code=404, detail="Incident introuvable")


@app.post("/api/emergency/evidence")
async def upload_evidence(file: UploadFile = File(...)):
    name = f"{int(time.time())}_{file.filename}"
    dest = EVIDENCE / name
    with dest.open("wb") as f:
        f.write(await file.read())
    return {"ok": True, "name": name, "url": f"/evidence/{name}"}


@app.post("/api/privacy/lock")
def set_privacy(p: PrivacyIn):
    state = _load(PRIVACY_JSON, {"enabled": False, "locked_at": None})
    state["enabled"] = p.enabled
    state["locked_at"] = _now_iso() if p.enabled else None
    _save(PRIVACY_JSON, state)
    return state

@app.get("/api/privacy/status")
def privacy_status():
    return _load(PRIVACY_JSON, {"enabled": False, "locked_at": None})
