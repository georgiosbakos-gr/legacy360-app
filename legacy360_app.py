# =========================================================
# Legacy360° V1 — Streamlit single-file app (FINAL w/ ADMIN INBOX)
# =========================================================
# Participant wizard (token invites)
# Admin dashboard (cases / invites / aggregation / inbox)
# Supabase backend (Postgres + JSONB via RPC)
#
# Admin Inbox:
# - Every submission creates an unread inbox item
# - Admin sees unread/read submissions without email or Edge Functions
# =========================================================

import os
import json
import hashlib
import secrets
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================================================
# APP CONFIG
# =========================================================

APP_VERSION = "2026-01-03-V1-INBOX"
QUESTIONNAIRE_VERSION = "v1"

st.set_page_config(page_title="Legacy360°", layout="wide")

params = st.query_params
is_admin = str(params.get("admin", "")).lower() in ("1", "true", "yes")
token = str(params.get("token", "")).strip()

# =========================================================
# SECRETS / SUPABASE
# =========================================================

def _get_secret(name: str, required: bool = True) -> str:
    v = ""
    try:
        v = str(st.secrets.get(name, "")).strip()
    except Exception:
        pass
    if not v:
        v = os.getenv(name, "").strip()
    if required and not v:
        raise RuntimeError(f"Missing secret/env: {name}")
    return v

def supabase_client(service: bool = False) -> Client:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY" if service else "SUPABASE_ANON_KEY")
    return create_client(url, key)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

# =========================================================
# ASSETS
# =========================================================

BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LEGACY_LOGO = os.path.join(ASSETS_DIR, "legacy360.png")
STRATEGIZE_LOGO = os.path.join(ASSETS_DIR, "strategize.png")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

# =========================================================
# ADMIN INBOX — NEW
# =========================================================

def ensure_admin_inbox_row(sb: Client, submission_id: str):
    sb.table("admin_inbox").upsert(
        {"submission_id": submission_id, "seen": False},
        on_conflict="submission_id"
    ).execute()

def mark_admin_seen(sb: Client, submission_id: str, by: str = "admin"):
    sb.table("admin_inbox").upsert(
        {
            "submission_id": submission_id,
            "seen": True,
            "seen_at": datetime.now(timezone.utc).isoformat(),
            "seen_by": by,
        },
        on_conflict="submission_id"
    ).execute()

def admin_inbox(sb: Client):
    st.subheader("📥 Inbox — New Submissions")

    unread_only = st.checkbox("Unread only", True)
    days = st.selectbox("Period (days)", [1, 7, 30], index=1)

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    subs = (
        sb.table("submissions")
        .select("submission_id,case_id,participant_email,submitted_at,derived_json,profile_json")
        .gte("submitted_at", since)
        .order("submitted_at", desc=True)
        .execute()
        .data or []
    )

    if not subs:
        st.info("No submissions.")
        return

    df = pd.DataFrame(subs)

    inbox = (
        sb.table("admin_inbox")
        .select("submission_id,seen")
        .in_("submission_id", df["submission_id"].tolist())
        .execute()
        .data or []
    )

    df_in = pd.DataFrame(inbox) if inbox else pd.DataFrame(columns=["submission_id", "seen"])
    df = df.merge(df_in, on="submission_id", how="left")
    df["seen"] = df["seen"].fillna(False)

    if unread_only:
        df = df[df["seen"] == False]

    for _, r in df.iterrows():
        derived = r.get("derived_json") or {}
        overall = derived.get("overall")

        with st.container(border=True):
            st.write(f"**Case:** `{r['case_id']}`")
            st.write(f"**Participant:** {r['participant_email']}")
            st.write(f"**Submitted:** {r['submitted_at']}")
            st.write(f"**Overall score:** {overall}")

            if not r["seen"]:
                if st.button("Mark as read", key=r["submission_id"]):
                    mark_admin_seen(sb, r["submission_id"])
                    st.rerun()

# =========================================================
# DB — PARTICIPANT
# =========================================================

def db_validate_invite(raw_token: str) -> Dict[str, Any]:
    sb = supabase_client()
    h = sha256_hex(raw_token)
    res = sb.rpc("validate_invite", {"p_token_hash": h}).execute()
    return res.data[0] if res.data else {"valid": False}

def db_submit_assessment(raw_token: str, lang: str, answers, profile, derived):
    sb = supabase_client()
    h = sha256_hex(raw_token)

    res = sb.rpc(
        "submit_assessment",
        {
            "p_token_hash": h,
            "p_lang": lang,
            "p_questionnaire_version": QUESTIONNAIRE_VERSION,
            "p_answers": answers,
            "p_profile": profile,
            "p_derived": derived,
        }
    ).execute()

    # 👇 NEW: create admin inbox record
    submission_id = res.data[0]["submission_id"]
    sb_admin = supabase_client(service=True)
    ensure_admin_inbox_row(sb_admin, submission_id)

    return res.data

# =========================================================
# DB — ADMIN
# =========================================================

def db_cases():
    sb = supabase_client(True)
    return sb.table("cases").select("*").order("created_at", desc=True).execute().data or []

def db_create_case(payload):
    sb = supabase_client(True)
    return sb.table("cases").insert(payload).execute().data[0]["case_id"]

def db_create_invite(case_id, email, days=14):
    sb = supabase_client(True)
    raw = secrets.token_urlsafe(32)
    h = sha256_hex(raw)
    exp = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

    res = sb.table("invites").insert({
        "case_id": case_id,
        "participant_email": email,
        "token_hash": h,
        "token_expires_at": exp,
        "max_uses": 1,
        "uses_count": 0,
        "status": "ACTIVE",
    }).execute()

    return raw

# =========================================================
# UI — HEADER
# =========================================================

lang = st.sidebar.radio("Γλώσσα / Language", ["GR", "EN"], index=0)

def header():
    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        if os.path.exists(LEGACY_LOGO):
            st.image(LEGACY_LOGO, width=260)
        st.title("Legacy360°")
        st.caption("Family Governance & Succession Roadmap")
    with c2:
        if os.path.exists(STRATEGIZE_LOGO):
            st.image(STRATEGIZE_LOGO, width=220)
    st.caption(f"Build: {APP_VERSION}")
    st.divider()

# =========================================================
# ADMIN DASHBOARD
# =========================================================

def admin_dashboard():
    header()

    pw = st.text_input("Admin password", type="password")
    if pw != _get_secret("ADMIN_PASSWORD"):
        st.stop()

    sb = supabase_client(True)

    tabs = st.tabs(["📥 Inbox", "Cases", "Create Case", "Invites"])

    with tabs[0]:
        admin_inbox(sb)

    with tabs[1]:
        st.dataframe(pd.DataFrame(db_cases()), use_container_width=True)

    with tabs[2]:
        name = st.text_input("Company name")
        if st.button("Create") and name:
            cid = db_create_case({"company_name": name})
            st.success(f"Created case: {cid}")

    with tabs[3]:
        case_id = st.text_input("Case ID")
        email = st.text_input("Participant email")
        base = _get_secret("APP_BASE_URL", False)
        if st.button("Generate invite") and case_id and email:
            raw = db_create_invite(case_id, email)
            link = f"{base}/?token={raw}"
            st.code(link)

# =========================================================
# PARTICIPANT (wizard unchanged)
# =========================================================

def participant_app():
    header()
    if not token:
        st.error("Missing token")
        return

    v = db_validate_invite(token)
    if not v or not v.get("valid"):
        st.error("Invalid or expired invite")
        return

    st.success("Invite valid — questionnaire would run here")

# =========================================================
# ROUTER
# =========================================================

if is_admin:
    admin_dashboard()
else:
    participant_app()
