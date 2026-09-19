"""Beach water-quality collector (Vancouver Coastal Health).

VCH publishes only the LATEST sample per station — there is no public
history for the Metro Vancouver beaches — so this module polls the
reports and accumulates every (station, sample time) row in SQLite.

Sources
  * Route 1/2/3 PDFs  — E. coli + salinity + temperature per station.
  * Trout Lake PDF    — E. coli only (freshwater, different layout).
  * Geometric-means PDF — VCH's official 30-day geomean per beach.
  * HealthSpace (Lions Bay, Bowen Island) — HTML, E. coli only, but the
    page carries the full history back to 2014, so every poll backfills.

Sampling is weekly May–September; a single sample over 235 MPN triggers
a next-day resample, so off-cycle rows appear. Polling a few times a day
and de-duping on (station_id, sampled_at) catches both.

Each sample is enriched for spike analysis with prior 24/48/72 h rain
(Open-Meteo) and the Pt Atkinson tide at sample time (DFO IWLS). Cruise
ship calls live in their own table (imported — portvancouver.com sits
behind a bot challenge) and are joined by date at query time.

VCH serves 403 to non-browser User-Agents, hence the browser headers.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
import re
import statistics
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz
import requests

from . import db
from .envutil import getenv_ci

log = logging.getLogger("helper.wq")

VAN_TZ = pytz.timezone("America/Vancouver")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-CA,en;q=0.9",
}

ROUTE_PDFS = {
    "route1": "https://www.vch.ca/en/Documents/VCH-beach-route1.pdf",
    "route2": "https://www.vch.ca/en/document-library/VCH-beach-route2",
    "route3": "https://www.vch.ca/en/Documents/VCH-beach-route3.pdf",
}
TROUT_LAKE_PDF = "https://www.vch.ca/en/Documents/VCH-beach-TroutLake.pdf"
GMEAN_PDF = "https://www.vch.ca/en/Documents/VCH-beach-gmean.pdf"

_HS_BASE = "https://healthspace.ca"
_HS_NSF = _HS_BASE + "/Clients/VCHA/CoastGaribaldi/CoastGaribaldi_Website.nsf"
HEALTHSPACE_LISTS = {
    "Bowen Island": _HS_NSF + "/Water-Beaches-List-ByName?OpenView"
                    "&RestrictToCategory=A27AE18EA3651F718933F71CD9B02C80&Count=30",
    "Lions Bay": _HS_NSF + "/Water-Beaches-List-ByName?OpenView"
                 "&RestrictToCategory=3C7224CA69847BF569CFEB0BDFE55F94&Count=30",
}

# Guidelines for Canadian Recreational Water Quality (2023)
SINGLE_SAMPLE_LIMIT = 235     # beach action value, single sample
GMEAN_LIMIT = 200             # 30-day geometric mean

# Approximate beach positions — used ONLY to pick the rainfall grid cell.
# Keyed by lower-cased beach name; `gmean` is the label VCH uses for the
# same beach in the geometric-means report.
DEFAULT_LATLON = (49.29, -123.14)
BEACH_META = {
    "jericho beach": dict(lat=49.2727, lon=-123.1935, gmean="Jericho"),
    "locarno beach": dict(lat=49.2757, lon=-123.2090, gmean="Locarno"),
    "spanish banks": dict(lat=49.2772, lon=-123.2230, gmean="Spanish Banks"),
    "spanish banks extension": dict(lat=49.2775, lon=-123.2370, gmean="Spanish Banks Extension"),
    "acadia beach": dict(lat=49.2790, lon=-123.2440, gmean="Acadia Beach"),
    "wreck beach trail 6 (breakwater)": dict(lat=49.2620, lon=-123.2615, gmean="Trail 6, Breakwater"),
    "wreck beach trail 4 (towers)": dict(lat=49.2720, lon=-123.2590, gmean="Trail 4, Towers"),
    "iona beach": dict(lat=49.2190, lon=-123.2130, gmean="Iona"),
    "garry point": dict(lat=49.1245, lon=-123.1930, gmean="Garry Point"),
    "third beach": dict(lat=49.3035, lon=-123.1565, gmean="Third"),
    "second beach": dict(lat=49.2955, lon=-123.1510, gmean="Second"),
    "english bay beach": dict(lat=49.2867, lon=-123.1430, gmean="English Bay"),
    "sunset beach": dict(lat=49.2795, lon=-123.1385, gmean="Sunset"),
    "kitsilano beach": dict(lat=49.2745, lon=-123.1540, gmean="Kitsilano Beach"),
    "kitsilano point": dict(lat=49.2775, lon=-123.1500, gmean="Kitsilano Point"),
    "west false creek": dict(lat=49.2720, lon=-123.1330, gmean="West False Creek"),
    "central false creek": dict(lat=49.2690, lon=-123.1200, gmean="Central False Creek"),
    "east false creek": dict(lat=49.2735, lon=-123.1060, gmean="East False Creek"),
    "cates park": dict(lat=49.3020, lon=-122.9580, gmean="Cates Park"),
    "deep cove": dict(lat=49.3290, lon=-122.9480, gmean="Deep Cove"),
    "crab park": dict(lat=49.2850, lon=-123.1010, gmean="Crab Park"),
    "brockton point": dict(lat=49.3010, lon=-123.1170, gmean="Brockton Point"),
    "ambleside beach": dict(lat=49.3235, lon=-123.1500, gmean="Ambleside"),
    "dundarave": dict(lat=49.3335, lon=-123.1840, gmean="Dundarave"),
    "sandy cove": dict(lat=49.3390, lon=-123.2270, gmean="Sandy Cove"),
    "eagle harbour": dict(lat=49.3530, lon=-123.2700, gmean="Eagle Harbour"),
    "whytecliff park": dict(lat=49.3720, lon=-123.2900, gmean="Whytecliff Park"),
    "trout lake": dict(lat=49.2555, lon=-123.0625, gmean="Trout Lake"),
    "lions bay beach": dict(lat=49.4540, lon=-123.2380, gmean="Lions Bay Beach"),
    "kelvin grove beach": dict(lat=49.4480, lon=-123.2390, gmean="Kelvin Grove Beach"),
    "brunswick beach": dict(lat=49.4720, lon=-123.2440, gmean="Brunswick Beach"),
    "snug cove": dict(lat=49.3800, lon=-123.3330, gmean="Snug Cove"),
    "sandy beach": dict(lat=49.3795, lon=-123.3270, gmean="Sandy Beach"),
    "pebbly beach": dict(lat=49.3840, lon=-123.3290, gmean="Pebbly Beach"),
    "mothers beach": dict(lat=49.3830, lon=-123.3330, gmean="Mothers Beach"),
    "tunstall bay": dict(lat=49.3570, lon=-123.4200, gmean="Tunstall Bay"),
    "bowen bay": dict(lat=49.3700, lon=-123.4250, gmean="Bowen Bay"),
}



def _meta(beach: str) -> dict | None:
    """BEACH_META lookup tolerant of HealthSpace spellings
    ("Mother's Beach", "Bowen Bay Beach")."""
    name = (beach or "").lower().replace("'", "").replace("\u2019", "").strip()
    return BEACH_META.get(name) or BEACH_META.get(name.removesuffix(" beach").strip())


_init_lock = threading.Lock()
_initialized = False
_cycle_lock = threading.Lock()


# ── Storage ────────────────────────────────────────────────────────────

def init():
    global _initialized
    with _init_lock:
        if _initialized:
            return
        db.init()
        c = db._conn()
        c.execute("""
            CREATE TABLE IF NOT EXISTS wq_samples (
                station_id TEXT NOT NULL,
                sampled_at TEXT NOT NULL,       -- Vancouver local ISO, minute precision
                has_time INTEGER NOT NULL DEFAULT 1,
                beach_key TEXT NOT NULL,
                area TEXT, beach TEXT, station_label TEXT,
                ecoli_raw TEXT, ecoli_num REAL, ecoli_censor TEXT,
                salinity_ppt REAL, temp_c REAL,
                is_reference INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                rain_24h REAL, rain_48h REAL, rain_72h REAL,
                tide_m REAL, tide_dir TEXT,
                enrich_tries INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (station_id, sampled_at)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_wq_beach ON wq_samples(beach_key, sampled_at)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS wq_gmean (
                label TEXT NOT NULL, calculated_on TEXT NOT NULL, value REAL,
                first_seen TEXT NOT NULL,
                PRIMARY KEY (label, calculated_on)
            )
        """)
        c.execute("CREATE TABLE IF NOT EXISTS wq_stars (beach_key TEXT PRIMARY KEY)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS cruise_calls (
                date TEXT NOT NULL, ship TEXT NOT NULL,
                arrival TEXT, departure TEXT, berth TEXT, source TEXT,
                PRIMARY KEY (date, ship)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS wq_runs (
                source TEXT PRIMARY KEY, ran_at TEXT NOT NULL, ok INTEGER NOT NULL,
                new_rows INTEGER NOT NULL DEFAULT 0, message TEXT, content_hash TEXT,
                last_ok_at TEXT
            )
        """)
        empty = c.execute("SELECT COUNT(*) AS n FROM cruise_calls").fetchone()["n"] == 0
        c.close()
        _initialized = True
    # Canada Place calls captured from portvancouver.com (the site can't be
    # polled from a server). Later seasons arrive via import_cruise_csv().
    seed = Path(__file__).resolve().parent / "seed" / "cruise_calls.csv"
    if empty and seed.exists():
        log.info("[wq] seeded cruise calls: %s", import_cruise_csv(seed.read_text(), "seed"))


def _now_iso() -> str:
    return datetime.now(VAN_TZ).isoformat(timespec="seconds")


def _record_run(source, ok, new_rows=0, message="", content_hash=None):
    c = db._conn()
    try:
        now = _now_iso()
        c.execute("""
            INSERT INTO wq_runs(source, ran_at, ok, new_rows, message, content_hash, last_ok_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
              ran_at = excluded.ran_at, ok = excluded.ok, new_rows = excluded.new_rows,
              message = excluded.message,
              content_hash = COALESCE(excluded.content_hash, wq_runs.content_hash),
              last_ok_at = COALESCE(excluded.last_ok_at, wq_runs.last_ok_at)
        """, (source, now, 1 if ok else 0, new_rows, message[:500], content_hash,
              now if ok else None))
    finally:
        c.close()


def _last_run(source):
    c = db._conn()
    try:
        r = c.execute("SELECT * FROM wq_runs WHERE source=?", (source,)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()


def upsert_samples(rows: list[dict]) -> int:
    """Insert new samples; refresh the measured values of known ones (VCH
    occasionally corrects a report). Returns how many rows were new."""
    init()
    new = 0
    c = db._conn()
    try:
        for r in rows:
            exists = c.execute("SELECT 1 FROM wq_samples WHERE station_id=? AND sampled_at=?",
                               (r["station_id"], r["sampled_at"])).fetchone()
            if exists:
                c.execute("""UPDATE wq_samples SET ecoli_raw=?, ecoli_num=?, ecoli_censor=?,
                               salinity_ppt=?, temp_c=?, is_reference=?
                             WHERE station_id=? AND sampled_at=?""",
                          (r["ecoli_raw"], r["ecoli_num"], r["ecoli_censor"],
                           r.get("salinity_ppt"), r.get("temp_c"), r.get("is_reference", 0),
                           r["station_id"], r["sampled_at"]))
                continue
            c.execute("""INSERT INTO wq_samples(station_id, sampled_at, has_time, beach_key, area,
                           beach, station_label, ecoli_raw, ecoli_num, ecoli_censor, salinity_ppt,
                           temp_c, is_reference, source, first_seen)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (r["station_id"], r["sampled_at"], r.get("has_time", 1), r["beach_key"],
                       r["area"], r["beach"], r["station_label"], r["ecoli_raw"], r["ecoli_num"],
                       r["ecoli_censor"], r.get("salinity_ppt"), r.get("temp_c"),
                       r.get("is_reference", 0), r["source"], _now_iso()))
            new += 1
    finally:
        c.close()
    return new


# ── Parsing ────────────────────────────────────────────────────────────

def parse_ecoli(raw: str):
    """'<10' → (10, '<'); '>24196' → (24196, '>'); 'LT5' / 'L5' → (5, '<').
    Censored values are stored at their detection limit."""
    s = (raw or "").strip().upper().replace(",", "")
    if not s:
        return None, None
    censor = None
    m = re.match(r"^(<|>|LT|GT|L|G)\s*(\d+(?:\.\d+)?)$", s)
    if m:
        censor = "<" if m.group(1) in ("<", "LT", "L") else ">"
        s = m.group(2)
    try:
        return float(s), censor
    except ValueError:
        return None, None


def _beach_key(area: str, beach: str) -> str:
    return f"{area} | {beach}"


_ROUTE_ROW = re.compile(
    r"^(B[A-Z]{2}-\d{2}-\w+)\s+(.+?)\s+(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})\s+"
    r"([<>]?\s*\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


def parse_route_text(text: str, source: str) -> list[dict]:
    """Rows of a Route 1/2/3 report. Stations listed under 'Reference
    Locations' (False Creek, Garry Point, creek mouths) are flagged — VCH
    excludes them from advisories and geometric means."""
    out = []
    in_reference = False
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("reference locations"):
            in_reference = True
            continue
        m = _ROUTE_ROW.match(line)
        if not m:
            continue
        sid, desc, d, t, ecoli, sal, temp = m.groups()
        parts = [p.strip() for p in desc.split(",")]
        area = parts[0]
        beach = parts[1] if len(parts) > 1 else parts[0]
        label = parts[2] if len(parts) > 2 else sid
        dt = datetime.strptime(f"{d} {t}", "%d/%m/%Y %H:%M")
        num, censor = parse_ecoli(ecoli)
        out.append(dict(
            station_id=sid, sampled_at=dt.strftime("%Y-%m-%dT%H:%M"), has_time=1,
            beach_key=_beach_key(area, beach), area=area, beach=beach, station_label=label,
            ecoli_raw=ecoli.replace(" ", ""), ecoli_num=num, ecoli_censor=censor,
            salinity_ppt=float(sal), temp_c=float(temp),
            is_reference=1 if in_reference else 0, source=source))
    return out


_TROUT_ROW = re.compile(
    r"Trout Lake \((\w+)\)\s+([<>]?\s*\d+)\s+(\d{1,2}-[A-Za-z]{3}-\d{4})\s+(\d{1,2}:\d{2})")


def parse_trout_text(text: str) -> list[dict]:
    out = []
    for m in _TROUT_ROW.finditer(text):
        side, ecoli, d, t = m.groups()
        dt = datetime.strptime(f"{d} {t}", "%d-%b-%Y %H:%M")
        num, censor = parse_ecoli(ecoli)
        out.append(dict(
            station_id=f"TROUT-{side.upper()}", sampled_at=dt.strftime("%Y-%m-%dT%H:%M"),
            has_time=1, beach_key=_beach_key("Vancouver", "Trout Lake"), area="Vancouver",
            beach="Trout Lake", station_label=side.title(), ecoli_raw=ecoli.replace(" ", ""),
            ecoli_num=num, ecoli_censor=censor, is_reference=0, source="troutlake"))
    return out


_GMEAN_ROW = re.compile(
    r"^(.+?)\s+(\d+)\s+((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+\d{4})$")


def parse_gmean_text(text: str) -> list[dict]:
    """(label, value, calculated_on). The label keeps whatever area prefix
    the PDF's first column bled into the line — gmean_for_beach() matches
    on the label's tail, so the prefix is harmless."""
    out = []
    for line in text.splitlines():
        m = _GMEAN_ROW.match(line.strip())
        if not m:
            continue
        label, val, d = m.groups()
        out.append(dict(label=label.strip(), value=float(val),
                        calculated_on=datetime.strptime(d, "%B %d, %Y").strftime("%Y-%m-%d")))
    return out


def _pdf_text(content: bytes) -> str:
    import pdfplumber   # heavy import — keep it off the app's startup path
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


# ── HealthSpace (Lions Bay / Bowen Island) ─────────────────────────────

def healthspace_insecure_tls() -> bool:
    """healthspace.ca has been serving an EXPIRED certificate (since
    2026-08-24). Collecting from it means skipping verification, so it is
    opt-in: WQ_HEALTHSPACE_INSECURE_TLS=1. Data is public and read-only."""
    return (getenv_ci("WQ_HEALTHSPACE_INSECURE_TLS", "0") or "0").lower() in ("1", "true", "yes")


def _hs_get(url: str, referer: str | None = None) -> str:
    headers = dict(_HEADERS)
    if referer:
        headers["Referer"] = referer    # the sample views 403 without it
    try:
        r = requests.get(url, headers=headers, timeout=40)
    except requests.exceptions.SSLError:
        if not healthspace_insecure_tls():
            raise
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, headers=headers, timeout=40, verify=False)
    r.raise_for_status()
    return r.text


def parse_healthspace_samples(html: str, area: str, beach: str) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out = []
    ecoli_col = None
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        lowered = [c.lower().replace(".", "").replace(" ", "") for c in cells]
        if "ecoli" in lowered:
            ecoli_col = lowered.index("ecoli")
            continue
        if ecoli_col is None or len(cells) <= ecoli_col:
            continue
        try:
            d = datetime.strptime(cells[0], "%d-%b-%Y")
        except ValueError:
            continue
        if d.year < 2000:       # placeholder rows dated 1800-01-01
            continue
        raw = cells[ecoli_col]
        num, censor = parse_ecoli(raw)
        if num is None:
            continue
        site = cells[1] or beach
        sid = "HS-" + re.sub(r"[^A-Z0-9]+", "-", site.upper()).strip("-")
        out.append(dict(
            station_id=sid, sampled_at=d.strftime("%Y-%m-%dT00:00"), has_time=0,
            beach_key=_beach_key(area, beach), area=area, beach=beach, station_label=site,
            ecoli_raw=raw, ecoli_num=num, ecoli_censor=censor, is_reference=0,
            source="healthspace"))
    return out


def fetch_healthspace() -> int:
    from bs4 import BeautifulSoup
    new = 0
    for area, list_url in HEALTHSPACE_LISTS.items():
        soup = BeautifulSoup(_hs_get(list_url), "html.parser")
        for a in soup.find_all("a", href=True):
            if "Water-Beaches-Samples" not in a["href"]:
                continue
            name = a.get_text(" ", strip=True)
            beach = name.split(",", 1)[1].strip() if "," in name else name
            html = _hs_get(_HS_BASE + a["href"].replace("&amp;", "&"), referer=list_url)
            new += upsert_samples(parse_healthspace_samples(html, area, beach))
            time.sleep(1)       # small, old Domino server — be gentle
    return new


# ── Fetch cycle ────────────────────────────────────────────────────────

def _archive_pdf(name: str, content: bytes):
    """Keep every distinct report so history can be re-parsed if the
    layout (or this parser) changes."""
    d = db.DB_PATH.parent / "wq_pdfs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}-{datetime.now(VAN_TZ).strftime('%Y%m%d-%H%M')}.pdf").write_bytes(content)


def _fetch_pdf_source(name: str, url: str, handler) -> None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=40)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError(f"not a PDF ({r.headers.get('content-type')})")
        digest = hashlib.sha1(r.content).hexdigest()
        prev = _last_run(name)
        if prev and prev.get("content_hash") == digest and prev.get("ok"):
            _record_run(name, True, 0, "unchanged", digest)
            return
        new, parsed = handler(_pdf_text(r.content))
        if not parsed:
            raise ValueError("report parsed to 0 rows — layout change?")
        _archive_pdf(name, r.content)
        _record_run(name, True, new, f"{parsed} rows parsed, {new} new", digest)
        log.info("[wq] %s: %d rows, %d new", name, parsed, new)
    except Exception as e:
        log.exception("[wq] %s failed", name)
        _record_run(name, False, 0, f"{type(e).__name__}: {e}")


def _store_gmean(rows: list[dict]) -> int:
    c = db._conn()
    new = 0
    try:
        for g in rows:
            cur = c.execute("""INSERT OR IGNORE INTO wq_gmean(label, calculated_on, value, first_seen)
                               VALUES(?,?,?,?)""",
                            (g["label"], g["calculated_on"], g["value"], _now_iso()))
            new += cur.rowcount
    finally:
        c.close()
    return new


def run_cycle(force_healthspace: bool = False) -> dict:
    """Poll every source, then enrich whatever is new. Safe to call often:
    unchanged PDFs are skipped by content hash."""
    init()
    if not _cycle_lock.acquire(blocking=False):
        return {"ok": False, "message": "cycle already running"}
    try:
        for name, url in ROUTE_PDFS.items():
            def handle(text, _n=name):
                rows = parse_route_text(text, _n)
                return upsert_samples(rows), len(rows)
            _fetch_pdf_source(name, url, handle)

        def handle_trout(text):
            rows = parse_trout_text(text)
            return upsert_samples(rows), len(rows)
        _fetch_pdf_source("troutlake", TROUT_LAKE_PDF, handle_trout)

        def handle_gmean(text):
            rows = parse_gmean_text(text)
            return _store_gmean(rows), len(rows)
        _fetch_pdf_source("gmean", GMEAN_PDF, handle_gmean)

        # HealthSpace returns full history each time — once a day is plenty.
        prev = _last_run("healthspace")
        due = True
        if prev and prev.get("last_ok_at") and not force_healthspace:
            age = datetime.now(VAN_TZ) - datetime.fromisoformat(prev["last_ok_at"])
            due = age > timedelta(hours=20)
        if due:
            try:
                n = fetch_healthspace()
                _record_run("healthspace", True, n, f"{n} new")
            except requests.exceptions.SSLError:
                _record_run("healthspace", False, 0,
                            "TLS certificate invalid (expired) — set "
                            "WQ_HEALTHSPACE_INSECURE_TLS=1 to collect anyway")
            except Exception as e:
                log.exception("[wq] healthspace failed")
                _record_run("healthspace", False, 0, f"{type(e).__name__}: {e}")

        try:
            enrich_pending()
        except Exception:
            log.exception("[wq] enrichment failed")
        return {"ok": True, "status": status()}
    finally:
        _cycle_lock.release()


# ── Enrichment: rain + tide ────────────────────────────────────────────

ENRICH_MAX_AGE_DAYS = 800     # graphs show 2 years; older rows stay bare
ENRICH_MAX_TRIES = 5
_IWLS = "https://api-iwls.dfo-mpo.gc.ca/api/v1"
_PT_ATKINSON = "5cebf1de3d0f4a073c4bb94c"


def _latlon(beach: str):
    m = _meta(beach)
    return (m["lat"], m["lon"]) if m else DEFAULT_LATLON


def _sample_dt(row) -> datetime:
    dt = datetime.fromisoformat(row["sampled_at"])
    # HealthSpace gives a date only; samplers work mornings
    return dt if row["has_time"] else dt.replace(hour=9)


RAIN_RECENT_DAYS = 45


def _rain_hourly(lat, lon, start: datetime, end: datetime, recent: bool) -> dict:
    """{local 'YYYY-MM-DDTHH:00': mm}. The forecast endpoint only has the
    last ~10 weeks (older hours come back null); the ERA5 archive lags
    ≈5 days but covers everything before that."""
    base = ("https://api.open-meteo.com/v1/forecast" if recent
            else "https://archive-api.open-meteo.com/v1/archive")
    r = requests.get(base, params=dict(
        latitude=lat, longitude=lon, hourly="precipitation",
        start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"),
        timezone="America/Vancouver"), timeout=60)
    r.raise_for_status()
    h = r.json().get("hourly", {})
    return {t: v for t, v in zip(h.get("time", []), h.get("precipitation", []))}


def _rain_sums(series: dict, at: datetime):
    out = []
    top = at.replace(minute=0)
    for hours in (24, 48, 72):
        vals = [series.get((top - timedelta(hours=i)).strftime("%Y-%m-%dT%H:00"))
                for i in range(hours)]
        if any(v is None for v in vals):
            return None
        out.append(round(sum(vals), 1))
    return out


def _tide_day(day: datetime):
    """Pt Atkinson predicted water level for one local day, 15-min steps."""
    start = VAN_TZ.localize(day.replace(hour=0, minute=0)).astimezone(pytz.utc)
    end = start + timedelta(hours=25)
    r = requests.get(f"{_IWLS}/stations/{_PT_ATKINSON}/data", params={
        "time-series-code": "wlp", "resolution": "FIFTEEN_MINUTES",
        "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=40)
    r.raise_for_status()
    return [(datetime.fromisoformat(p["eventDate"].replace("Z", "+00:00")), p["value"])
            for p in r.json()]


def _tide_at(points, at_local: datetime):
    if len(points) < 2:
        return None, None
    at = VAN_TZ.localize(at_local).astimezone(pytz.utc)
    i = min(range(len(points)), key=lambda k: abs(points[k][0] - at))
    a, b = (i, i + 1) if i + 1 < len(points) else (i - 1, i)
    return round(points[i][1], 2), ("rising" if points[b][1] > points[a][1] else "falling")


def enrich_pending() -> int:
    init()
    cutoff = (datetime.now(VAN_TZ) - timedelta(days=ENRICH_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    c = db._conn()
    try:
        rows = [dict(r) for r in c.execute("""
            SELECT station_id, sampled_at, has_time, beach, rain_72h, tide_m FROM wq_samples
            WHERE sampled_at >= ? AND enrich_tries < ?
              AND (rain_72h IS NULL OR (tide_m IS NULL AND has_time = 1))""",
            (cutoff, ENRICH_MAX_TRIES))]
    finally:
        c.close()
    if not rows:
        return 0

    # Rain: one request per ~10 km grid cell and era (recent / archive)
    recent_floor = datetime.now(VAN_TZ).replace(tzinfo=None) - timedelta(days=RAIN_RECENT_DAYS)
    groups: dict = {}
    for r in rows:
        if r["rain_72h"] is not None:
            continue
        lat, lon = _latlon(r["beach"])
        at = _sample_dt(r)
        groups.setdefault((round(lat, 1), round(lon, 1), at >= recent_floor), []).append(r)
    for (lat, lon, recent), members in groups.items():
        times = [_sample_dt(m) for m in members]
        try:
            series = _rain_hourly(lat, lon, min(times) - timedelta(days=4), max(times), recent)
        except Exception as e:
            log.warning("[wq] rain fetch failed for %s,%s: %s", lat, lon, e)
            continue
        for m in members:
            m["_rain"] = _rain_sums(series, _sample_dt(m))
            m["_attempted"] = True
        time.sleep(1)

    # Tide: one request per distinct sample day
    tide_cache: dict = {}
    for r in rows:
        if r["tide_m"] is not None or not r["has_time"]:
            continue
        at = _sample_dt(r)
        day = at.strftime("%Y-%m-%d")
        if day not in tide_cache:
            try:
                tide_cache[day] = _tide_day(at)
            except Exception as e:
                log.warning("[wq] tide fetch failed for %s: %s", day, e)
                tide_cache[day] = None
        if tide_cache[day] is not None:
            r["_tide"] = _tide_at(tide_cache[day], at)
            r["_attempted"] = True

    done = 0
    c = db._conn()
    try:
        for r in rows:
            rain = r.get("_rain")
            tide = r.get("_tide") or (None, None)
            c.execute("""UPDATE wq_samples SET
                           rain_24h = COALESCE(?, rain_24h), rain_48h = COALESCE(?, rain_48h),
                           rain_72h = COALESCE(?, rain_72h), tide_m = COALESCE(?, tide_m),
                           tide_dir = COALESCE(?, tide_dir), enrich_tries = enrich_tries + ?
                         WHERE station_id=? AND sampled_at=?""",
                      (*(rain or (None, None, None)), tide[0], tide[1],
                       1 if r.get("_attempted") else 0,     # upstream outages don't burn tries
                       r["station_id"], r["sampled_at"]))
            done += 1 if (rain or tide[0] is not None) else 0
    finally:
        c.close()
    log.info("[wq] enriched %d/%d samples", done, len(rows))
    return done


# ── Cruise ship calls ──────────────────────────────────────────────────

def _norm_date(s: str) -> str | None:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d/%m/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def import_cruise_csv(text: str, source: str = "import") -> dict:
    """CSV with a header row: date, ship[, arrival, departure, berth].
    Dates may be ISO or 'Sep 18, 2026' (as shown on portvancouver.com)."""
    init()
    reader = csv.DictReader(io.StringIO(text.strip()))
    reader.fieldnames = [(f or "").strip().lower() for f in (reader.fieldnames or [])]
    added = skipped = 0
    c = db._conn()
    try:
        for row in reader:
            d = _norm_date(row.get("date", ""))
            ship = (row.get("ship") or "").strip()
            if not d or not ship:
                skipped += 1
                continue
            c.execute("""INSERT INTO cruise_calls(date, ship, arrival, departure, berth, source)
                         VALUES(?,?,?,?,?,?)
                         ON CONFLICT(date, ship) DO UPDATE SET arrival=excluded.arrival,
                           departure=excluded.departure, berth=excluded.berth""",
                      (d, ship, (row.get("arrival") or "").strip(),
                       (row.get("departure") or "").strip(),
                       (row.get("berth") or "").strip(), source))
            added += 1
    finally:
        c.close()
    return {"imported": added, "skipped": skipped}


def cruise_calls(start: str, end: str) -> list[dict]:
    init()
    c = db._conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT date, ship, arrival, departure, berth FROM cruise_calls "
            "WHERE date BETWEEN ? AND ? ORDER BY date, arrival", (start, end))]
    finally:
        c.close()


def _cruise_counts(start: str, end: str) -> dict:
    counts: dict = {}
    for call in cruise_calls(start, end):
        counts[call["date"]] = counts.get(call["date"], 0) + 1
    return counts


# ── Queries ────────────────────────────────────────────────────────────

def set_star(beach_key: str, starred: bool):
    init()
    c = db._conn()
    try:
        if starred:
            c.execute("INSERT OR IGNORE INTO wq_stars(beach_key) VALUES(?)", (beach_key,))
        else:
            c.execute("DELETE FROM wq_stars WHERE beach_key=?", (beach_key,))
    finally:
        c.close()


def _geomean(vals):
    vals = [v for v in vals if v and v > 0]
    return round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 1) if vals else None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def gmean_for_beach(beach: str, latest: dict):
    """Match VCH's geometric-mean label to a beach. Labels may carry an
    area prefix ('West Vancouver Ambleside'), so match on the tail."""
    alias = (_meta(beach) or {}).get("gmean") or beach
    best = None
    for label, g in latest.items():
        if label == alias or label.endswith(" " + alias):
            if best is None or len(label) < len(best[0]):
                best = (label, g)
    return best[1] if best else None


def _spike(latest_num, prior_nums):
    """A reading is a spike when it breaches the single-sample limit, or
    jumps to ≥3× the station's trailing median (and is not just noise
    near the detection limit)."""
    if latest_num is None:
        return None
    if latest_num >= SINGLE_SAMPLE_LIMIT:
        return "over-limit"
    if len(prior_nums) >= 4:
        med = statistics.median(prior_nums)
        if latest_num >= 50 and latest_num >= 3 * max(med, 10):
            return "jump"
    return None


def latest() -> dict:
    """Current values per beach (newest sample per station), stars first."""
    init()
    c = db._conn()
    try:
        rows = [dict(r) for r in c.execute("""
            SELECT s.* FROM wq_samples s
            JOIN (SELECT station_id, MAX(sampled_at) AS m FROM wq_samples GROUP BY station_id) t
              ON t.station_id = s.station_id AND t.m = s.sampled_at""")]
        stars = {r["beach_key"] for r in c.execute("SELECT beach_key FROM wq_stars")}
        gm_latest = {r["label"]: dict(r) for r in c.execute("""
            SELECT g.label, g.value, g.calculated_on FROM wq_gmean g
            JOIN (SELECT label, MAX(calculated_on) AS m FROM wq_gmean GROUP BY label) t
              ON t.label = g.label AND t.m = g.calculated_on""")}
        prior: dict = {}
        for r in c.execute("""SELECT station_id, sampled_at, ecoli_num FROM wq_samples
                              WHERE ecoli_num IS NOT NULL ORDER BY sampled_at DESC"""):
            lst = prior.setdefault(r["station_id"], [])
            if len(lst) < 9:
                lst.append(r["ecoli_num"])
    finally:
        c.close()

    beaches: dict = {}
    for r in rows:
        b = beaches.setdefault(r["beach_key"], dict(
            key=r["beach_key"], area=r["area"], beach=r["beach"], source=r["source"],
            stations=[]))
        r["spike"] = _spike(r["ecoli_num"], prior.get(r["station_id"], [])[1:])
        b["stations"].append({k: r[k] for k in (
            "station_id", "station_label", "sampled_at", "has_time", "ecoli_raw", "ecoli_num",
            "ecoli_censor", "salinity_ppt", "temp_c", "is_reference", "rain_24h", "rain_48h",
            "rain_72h", "tide_m", "tide_dir", "spike")})

    today = datetime.now(VAN_TZ).date()
    cruise = _cruise_counts((today - timedelta(days=400)).isoformat(), today.isoformat())
    out = []
    for b in beaches.values():
        b["stations"].sort(key=lambda s: s["station_id"])
        # A beach's "current" reading = its most recent sampling day. Stations
        # retired years ago (HealthSpace) must not drag old values in.
        newest = max(s["sampled_at"] for s in b["stations"])
        current = [s for s in b["stations"] if s["sampled_at"][:10] == newest[:10]]
        # Reference-only sites still count when the whole beach is reference
        primary = [s for s in current if not s["is_reference"]] or current
        worst = max(primary, key=lambda s: (s["ecoli_num"] or 0, s["ecoli_censor"] == ">"))
        gm = gmean_for_beach(b["beach"], gm_latest)
        b.update(
            starred=b["key"] in stars,
            is_reference=all(s["is_reference"] for s in b["stations"]),
            sampled_at=newest,
            age_days=(today - datetime.fromisoformat(newest).date()).days,
            ecoli_worst_raw=worst["ecoli_raw"], ecoli_worst_num=worst["ecoli_num"],
            ecoli_geomean=_geomean([s["ecoli_num"] for s in primary]),
            salinity_ppt=_mean([s["salinity_ppt"] for s in primary]),
            temp_c=_mean([s["temp_c"] for s in primary]),
            rain_72h=_mean([s["rain_72h"] for s in primary]),
            tide_m=_mean([s["tide_m"] for s in primary]),
            tide_dir=next((s["tide_dir"] for s in primary if s["tide_dir"]), None),
            vch_gmean=gm["value"] if gm else None,
            vch_gmean_on=gm["calculated_on"] if gm else None,
            spike=next((s["spike"] for s in sorted(
                primary, key=lambda s: s["spike"] != "over-limit") if s["spike"]), None),
            cruise_ships_day=cruise.get(newest[:10], 0),
            current_station_ids=[s["station_id"] for s in current],
        )
        out.append(b)
    out.sort(key=lambda b: (not b["starred"], b["is_reference"], b["area"], b["beach"]))
    return {"beaches": out, "limits": {"single": SINGLE_SAMPLE_LIMIT, "gmean": GMEAN_LIMIT},
            "generated_at": _now_iso()}


def history(beach_key: str, days: int = 730) -> dict:
    init()
    since = (datetime.now(VAN_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
    c = db._conn()
    try:
        rows = [dict(r) for r in c.execute("""
            SELECT station_id, station_label, sampled_at, has_time, ecoli_raw, ecoli_num,
                   ecoli_censor, salinity_ppt, temp_c, is_reference, rain_24h, rain_48h,
                   rain_72h, tide_m, tide_dir, source
            FROM wq_samples WHERE beach_key=? AND sampled_at >= ? ORDER BY sampled_at""",
            (beach_key, since))]
        beach = beach_key.split(" | ", 1)[-1]
        gm_all = [dict(r) for r in c.execute(
            "SELECT label, value, calculated_on FROM wq_gmean WHERE calculated_on >= ? "
            "ORDER BY calculated_on", (since,))]
    finally:
        c.close()
    alias = (_meta(beach) or {}).get("gmean") or beach
    labels = {g["label"] for g in gm_all if g["label"] == alias or g["label"].endswith(" " + alias)}
    label = min(labels, key=len) if labels else None
    today = datetime.now(VAN_TZ).date().isoformat()
    return {
        "beach_key": beach_key, "days": days, "samples": rows,
        "vch_gmean": [g for g in gm_all if g["label"] == label],
        "cruise_calls": cruise_calls(since, today),
        "limits": {"single": SINGLE_SAMPLE_LIMIT, "gmean": GMEAN_LIMIT},
    }


CSV_COLUMNS = ["beach_key", "area", "beach", "station_id", "station_label", "sampled_at",
               "has_time", "ecoli_raw", "ecoli_num", "ecoli_censor", "salinity_ppt", "temp_c",
               "is_reference", "rain_24h", "rain_48h", "rain_72h", "tide_m", "tide_dir",
               "cruise_ships_day", "cruise_ships_prev_day", "source", "first_seen"]


def export_csv(beach_key: str | None = None, days: int = 730) -> str:
    init()
    since = (datetime.now(VAN_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
    q = "SELECT * FROM wq_samples WHERE sampled_at >= ?"
    args: list = [since]
    if beach_key:
        q += " AND beach_key = ?"
        args.append(beach_key)
    c = db._conn()
    try:
        rows = [dict(r) for r in c.execute(q + " ORDER BY beach_key, station_id, sampled_at", args)]
    finally:
        c.close()
    cruise = _cruise_counts(since, datetime.now(VAN_TZ).date().isoformat())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        day = datetime.fromisoformat(r["sampled_at"]).date()
        r["cruise_ships_day"] = cruise.get(day.isoformat(), 0)
        r["cruise_ships_prev_day"] = cruise.get((day - timedelta(days=1)).isoformat(), 0)
        w.writerow(r)
    return buf.getvalue()


def status() -> dict:
    init()
    c = db._conn()
    try:
        runs = [dict(r) for r in c.execute("SELECT * FROM wq_runs ORDER BY source")]
        tot = c.execute("""SELECT COUNT(*) AS n, COUNT(DISTINCT station_id) AS stations,
                                  COUNT(DISTINCT beach_key) AS beaches, MIN(sampled_at) AS first,
                                  MAX(sampled_at) AS last FROM wq_samples""").fetchone()
        n_cruise = c.execute("SELECT COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last "
                             "FROM cruise_calls").fetchone()
    finally:
        c.close()
    return {"runs": runs, "samples": dict(tot), "cruise_calls": dict(n_cruise),
            "healthspace_insecure_tls": healthspace_insecure_tls()}
