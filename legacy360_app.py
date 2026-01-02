# legacy360_app.py
# Legacy360° V1 — Streamlit single-file app
# Participant wizard (token invites) + Admin dashboard (cases/invites/aggregation) + Premium PDFs
# Supabase backend (Postgres + JSONB via RPC)

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.utils import ImageReader

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# CONFIG / SECRETS
# =========================================================

QUESTIONNAIRE_VERSION = "v1"

def _get_secret(name: str, required: bool = True) -> str:
    v = ""
    try:
        v = str(st.secrets.get(name, "")).strip()
    except Exception:
        v = ""
    if not v:
        v = os.getenv(name, "").strip()
    if required and not v:
        raise RuntimeError(f"Missing secret/env: {name}")
    return v

def supabase_client(use_service_role: bool = False) -> Client:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_SERVICE_ROLE_KEY" if use_service_role else "SUPABASE_ANON_KEY")
    return create_client(url, key)

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# =========================================================
# PDF FONTS (Greek-safe)
# =========================================================

def register_pdf_fonts():
    """
    Registers DejaVu fonts if available.
    Never crashes the app; falls back to default fonts if missing.
    Also supports both assets/fonts and assets/text (legacy path).
    """
    if getattr(register_pdf_fonts, "_done", False):
        return

    base = os.path.dirname(__file__)

    # Support both possible folders (in case files ended up in assets/text)
    candidate_dirs = [
        os.path.join(base, "assets", "fonts"),
        os.path.join(base, "assets", "text"),
    ]

    found_dir = None
    regular = None
    bold = None

    for d in candidate_dirs:
        r = os.path.join(d, "DejaVuSans.ttf")
        b = os.path.join(d, "DejaVuSans-Bold.ttf")
        if os.path.exists(r) and os.path.exists(b):
            found_dir = d
            regular = r
            bold = b
            break

    # Lightweight debug in sidebar (won't crash)
    try:
        st.sidebar.caption("PDF fonts:")
        st.sidebar.write("Font dir used:", found_dir)
        for d in candidate_dirs:
            st.sidebar.write("Dir exists:", d, os.path.exists(d))
            if os.path.exists(d):
                st.sidebar.write("Files:", os.listdir(d))
    except Exception:
        pass

    if regular and bold:
        pdfmetrics.registerFont(TTFont("DejaVu", regular))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))

    register_pdf_fonts._done = True



# =========================================================
# DOMAIN MODEL
# =========================================================

@dataclass
class Domain:
    key: str
    weight: float

@dataclass
class Question:
    id: str
    domain_key: str
    text: Dict[str, str]  # {"GR": "...", "EN": "..."}

DOMAINS: List[Domain] = [
    Domain("corp_gov", 0.20),
    Domain("family_gov", 0.20),
    Domain("family_roles", 0.15),
    Domain("strategy", 0.20),
    Domain("fin_perf", 0.15),
    Domain("sust_cont", 0.10),
]

DOMAIN_LABELS = {
    "GR": {
        "corp_gov": "Εταιρική Διακυβέρνηση",
        "family_gov": "Οικογενειακή Διακυβέρνηση",
        "family_roles": "Ρόλοι Μελών Οικογένειας στην Επιχείρηση",
        "strategy": "Στρατηγική Σαφήνεια",
        "fin_perf": "Χρηματοοικονομική & Επιχειρησιακή Διαφάνεια",
        "sust_cont": "Βιωσιμότητα & Συνέχεια",
    },
    "EN": {
        "corp_gov": "Corporate Governance",
        "family_gov": "Family Governance",
        "family_roles": "Roles of Family Members in the Business",
        "strategy": "Strategic Clarity",
        "fin_perf": "Financial & Performance Visibility",
        "sust_cont": "Sustainability & Continuity",
    },
}

QUESTIONS: List[Question] = [
    # Corporate Governance
    Question("1.1", "corp_gov", {"EN": "Roles and responsibilities of Board, Management and Shareholders are clearly defined and respected in practice.",
                                "GR": "Οι ρόλοι και οι αρμοδιότητες του Δ.Σ., της Διοίκησης και των Μετόχων είναι σαφώς καθορισμένοι και γίνονται σεβαστοί στην πράξη."}),
    Question("1.2", "corp_gov", {"EN": "The Board provides effective strategic oversight and constructive challenge.",
                                "GR": "Το Δ.Σ. ασκεί ουσιαστικό στρατηγικό έλεγχο και εποικοδομητική κριτική."}),
    Question("1.3", "corp_gov", {"EN": "Decision rights and escalation mechanisms are clear and consistently applied.",
                                "GR": "Τα decision rights και οι μηχανισμοί κλιμάκωσης είναι σαφείς και εφαρμόζονται με συνέπεια."}),
    Question("1.4", "corp_gov", {"EN": "Governance supports accountability, transparency and long-term value creation.",
                                "GR": "Η διακυβέρνηση υποστηρίζει λογοδοσία, διαφάνεια και μακροπρόθεσμη δημιουργία αξίας."}),

    # Family Governance
    Question("2.1", "family_gov", {"EN": "Family–Ownership–Business relationship is formally structured and governed.",
                                  "GR": "Η σχέση Οικογένειας–Ιδιοκτησίας–Επιχείρησης είναι δομημένη και τυπικά ορισμένη."}),
    Question("2.2", "family_gov", {"EN": "There are forums/processes for family alignment and conflict resolution.",
                                  "GR": "Υπάρχουν διαδικασίες για ευθυγράμμιση και επίλυση συγκρούσεων εντός της οικογένειας."}),
    Question("2.3", "family_gov", {"EN": "Family policies (employment/dividends/transfers) are defined and applied consistently.",
                                  "GR": "Οι οικογενειακές πολιτικές (απασχόληση/μερίσματα/μεταβιβάσεις) είναι ορισμένες και εφαρμόζονται με συνέπεια."}),
    Question("2.4", "family_gov", {"EN": "Family involvement supports continuity rather than creating governance risk.",
                                  "GR": "Η εμπλοκή της οικογένειας υποστηρίζει τη συνέχεια και δεν δημιουργεί κίνδυνο διακυβέρνησης."}),

    # Family roles
    Question("3.1", "family_roles", {"EN": "Roles and responsibilities of family members in the business are documented.",
                                    "GR": "Οι ρόλοι των μελών οικογένειας στην επιχείρηση είναι τεκμηριωμένοι."}),
    Question("3.2", "family_roles", {"EN": "Entry/progression/exit criteria for family members are objective and transparent.",
                                    "GR": "Τα κριτήρια εισόδου/εξέλιξης/εξόδου είναι αντικειμενικά και διαφανή."}),
    Question("3.3", "family_roles", {"EN": "Performance evaluation uses the same standards as for non-family executives.",
                                    "GR": "Η αξιολόγηση απόδοσης χρησιμοποιεί τα ίδια κριτήρια με τα μη οικογενειακά στελέχη."}),
    Question("3.4", "family_roles", {"EN": "Family roles add measurable value and do not rely on informal authority.",
                                    "GR": "Οι οικογενειακοί ρόλοι προσθέτουν μετρήσιμη αξία και δεν βασίζονται σε άτυπη εξουσία."}),

    # Strategy
    Question("4.1", "strategy", {"EN": "There is a clear strategy understood across leadership levels.",
                                "GR": "Υπάρχει σαφής στρατηγική κατανοητή σε επίπεδα ηγεσίας."}),
    Question("4.2", "strategy", {"EN": "Strategic priorities are translated into objectives, initiatives and execution plans.",
                                "GR": "Οι προτεραιότητες μεταφράζονται σε στόχους, πρωτοβουλίες και σχέδια υλοποίησης."}),
    Question("4.3", "strategy", {"EN": "Strategic decisions reflect agreed priorities, not ad-hoc considerations.",
                                "GR": "Οι στρατηγικές αποφάσεις αντανακλούν συμφωνημένες προτεραιότητες, όχι αποσπασματικές επιλογές."}),
    Question("4.4", "strategy", {"EN": "Strategy balances performance, family expectations and continuity.",
                                "GR": "Η στρατηγική ισορροπεί απόδοση, προσδοκίες οικογένειας και συνέχεια."}),

    # Financial & performance visibility
    Question("5.1", "fin_perf", {"EN": "Financial/performance info is timely, reliable and decision-relevant.",
                                "GR": "Η πληροφόρηση απόδοσης είναι έγκαιρη, αξιόπιστη και χρήσιμη για αποφάσεις."}),
    Question("5.2", "fin_perf", {"EN": "KPIs are clearly defined and aligned with strategic priorities.",
                                "GR": "Τα KPIs είναι σαφώς ορισμένα και ευθυγραμμισμένα με στρατηγικές προτεραιότητες."}),
    Question("5.3", "fin_perf", {"EN": "Performance discussions focus on insight and forward actions.",
                                "GR": "Οι συζητήσεις απόδοσης εστιάζουν σε insights και μελλοντικές ενέργειες."}),
    Question("5.4", "fin_perf", {"EN": "Transparency supports accountability in management and ownership.",
                                "GR": "Η διαφάνεια υποστηρίζει λογοδοσία στη διοίκηση και την ιδιοκτησία."}),

    # Sustainability & continuity
    Question("6.1", "sust_cont", {"EN": "There is a realistic succession approach for key leadership and ownership roles.",
                                 "GR": "Υπάρχει ρεαλιστική προσέγγιση διαδοχής για κρίσιμους ρόλους ηγεσίας και ιδιοκτησίας."}),
    Question("6.2", "sust_cont", {"EN": "Long-term risks are actively identified and managed.",
                                 "GR": "Οι μακροπρόθεσμοι κίνδυνοι εντοπίζονται και διαχειρίζονται ενεργά."}),
    Question("6.3", "sust_cont", {"EN": "Leadership development and talent pipelines support future needs.",
                                 "GR": "Η ανάπτυξη ηγεσίας και ταλέντων υποστηρίζει μελλοντικές ανάγκες."}),
    Question("6.4", "sust_cont", {"EN": "Sustainability is integrated into strategic/governance decisions.",
                                 "GR": "Η βιωσιμότητα ενσωματώνεται σε στρατηγικές/διακυβερνητικές αποφάσεις."}),
]


# =========================================================
# UI COPY
# =========================================================

UI = {
    "GR": {
        "title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "missing_token": "Λείπει ή είναι άκυρο το invite token. Παρακαλώ χρησιμοποιήστε το link που λάβατε.",
        "token_invalid": "Το invite token δεν είναι έγκυρο/έχει λήξει/έχει χρησιμοποιηθεί.",
        "token_used_readonly": "Το invite έχει ήδη χρησιμοποιηθεί. Μπορείτε να δείτε/κατεβάσετε τα αποτελέσματα, αλλά όχι να κάνετε νέα υποβολή.",
        "profile": "Στοιχεία Συμμετέχοντα",
        "case": "Case ID",
        "progress": "Πρόοδος",
        "submit": "✅ Υποβολή / Submit",
        "submitted_ok": "Η υποβολή καταχωρήθηκε επιτυχώς.",
        "results": "Αποτελέσματα",
        "download_pdf": "Λήψη PDF",
        "download_case_pdf": "Λήψη Case PDF (Alignment)",
        "admin": "Admin",
        "admin_password": "Κωδικός",
        "admin_wrong": "Λάθος κωδικός.",
    },
    "EN": {
        "title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "missing_token": "Missing or invalid invite token. Please use the link you received.",
        "token_invalid": "Invite token is invalid/expired/used.",
        "token_used_readonly": "Invite already used. You can view/download results but cannot submit again.",
        "profile": "Participant Profile",
        "case": "Case ID",
        "progress": "Progress",
        "submit": "✅ Submit",
        "submitted_ok": "Submission stored successfully.",
        "results": "Results",
        "download_pdf": "Download PDF",
        "download_case_pdf": "Download Case PDF (Alignment)",
        "admin": "Admin",
        "admin_password": "Password",
        "admin_wrong": "Wrong password.",
    }
}

BANDS = [
    ("RED", 0.0, 2.5),
    ("AMBER", 2.5, 3.5),
    ("GREEN", 3.5, 5.01),
]

BAND_LABELS = {
    "GR": {"RED": "ΚΟΚΚΙΝΟ", "AMBER": "ΚΙΤΡΙΝΟ", "GREEN": "ΠΡΑΣΙΝΟ"},
    "EN": {"RED": "RED", "AMBER": "AMBER", "GREEN": "GREEN"},
}


# =========================================================
# HELPERS: SCORING / AGGREGATION
# =========================================================

def domain_questions_map() -> Dict[str, List[str]]:
    m = {d.key: [] for d in DOMAINS}
    for q in QUESTIONS:
        m[q.domain_key].append(q.id)
    return m

def band_for_score(score: float) -> str:
    for b, lo, hi in BANDS:
        if lo <= score < hi:
            return b
    return "AMBER"

def compute_domain_scores(answers: Dict[str, int]) -> Dict[str, float]:
    dq = domain_questions_map()
    scores = {}
    for dom, qids in dq.items():
        vals = [answers.get(qid) for qid in qids]
        if any(v is None for v in vals):
            scores[dom] = float("nan")
        else:
            scores[dom] = float(np.mean(vals))
    return scores

def weighted_index(domain_scores: Dict[str, float]) -> float:
    total = 0.0
    for d in DOMAINS:
        s = domain_scores.get(d.key, float("nan"))
        if np.isnan(s):
            return float("nan")
        total += s * d.weight
    return (total - 1.0) / 4.0 * 100.0  # 1..5 -> 0..100

def risk_priority(avg_score: float, weight: float) -> float:
    return (6.0 - avg_score) * weight

def build_domain_df(lang: str, domain_scores: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for d in DOMAINS:
        avg = domain_scores[d.key]
        rows.append({
            "domain_key": d.key,
            "domain": DOMAIN_LABELS[lang][d.key],
            "weight": d.weight,
            "avg_score": float(avg),
            "band": band_for_score(float(avg)),
            "risk": risk_priority(float(avg), d.weight),
        })
    return pd.DataFrame(rows).sort_values("risk", ascending=False)

def aggregate_case(lang: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    domains = [d.key for d in DOMAINS]
    dom_vals = {k: [] for k in domains}
    overall_vals = []

    for s in submissions:
        dj = s.get("derived_json") or {}
        if isinstance(dj, str):
            try:
                dj = json.loads(dj)
            except Exception:
                dj = {}
        ds = dj.get("domain_scores") or {}
        for k in domains:
            v = ds.get(k)
            if v is None:
                continue
            try:
                dom_vals[k].append(float(v))
            except Exception:
                pass
        if dj.get("overall") is not None:
            try:
                overall_vals.append(float(dj["overall"]))
            except Exception:
                pass

    dom_avg = {k: (float(np.mean(dom_vals[k])) if dom_vals[k] else float("nan")) for k in domains}
    dom_std = {k: (float(np.std(dom_vals[k], ddof=0)) if len(dom_vals[k]) >= 2 else 0.0) for k in domains}
    overall_avg = float(np.mean(overall_vals)) if overall_vals else float("nan")

    case_df = build_domain_df(lang, dom_avg)
    case_df["std"] = case_df["domain_key"].map(dom_std)

    return {
        "participants_n": len(submissions),
        "domain_avg": dom_avg,
        "domain_std": dom_std,
        "overall_avg": overall_avg,
        "case_df": case_df,
    }


# =========================================================
# CHARTS
# =========================================================

def make_radar(labels: List[str], values: List[float], title: str):
    r = values + [values[0]]
    theta = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill="toself"))
    fig.update_layout(
        showlegend=False,
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        margin=dict(l=30, r=30, t=50, b=30),
        title=title,
        height=380,
    )
    return fig


# =========================================================
# PDF HELPERS (logos + wrapping)
# =========================================================

def _img_contain(path: str, max_w_mm: float, max_h_mm: float):
    """
    CSS-like 'contain': keep aspect ratio, fit into max_w x max_h box.
    Avoids squeezing for square PNGs.
    """
    try:
        if not (path and os.path.exists(path)):
            return None
        ir = ImageReader(path)
        iw, ih = ir.getSize()
        if iw <= 0 or ih <= 0:
            return None
        box_w = max_w_mm * mm
        box_h = max_h_mm * mm
        scale = min(box_w / float(iw), box_h / float(ih))
        w = iw * scale
        h = ih * scale
        return Image(path, width=w, height=h)
    except Exception:
        return None

def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_participant_pdf(lang: str, df_domains: pd.DataFrame, overall_0_100: float,
                          answers_df: pd.DataFrame, legacy_logo_path: str, strategize_logo_path: str) -> bytes:
    register_pdf_fonts()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2C5D")
    gold = colors.HexColor("#C7922B")
    grey = colors.HexColor("#6B7280")

    base = ParagraphStyle("base", parent=styles["BodyText"], fontName="DejaVu", fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=base, fontName="DejaVu", fontSize=9, leading=12, textColor=grey)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="DejaVu-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=12, leading=14, textColor=navy, spaceAfter=6)

    L = {
        "GR": {"report": "Αναφορά Αποτελεσμάτων", "date": "Ημερομηνία", "page": "Σελίδα",
               "summary": "Σύνοψη ανά Ενότητα", "domain": "Ενότητα", "weight": "Βάρος", "score": "Βαθμός",
               "status": "Κατάσταση", "risk": "Κίνδυνος", "appendix": "Παράρτημα: Απαντήσεις"},
        "EN": {"report": "Results Report", "date": "Date", "page": "Page",
               "summary": "Domain Summary", "domain": "Domain", "weight": "Weight", "score": "Score",
               "status": "Status", "risk": "Risk", "appendix": "Appendix: Responses"},
    }[lang]

    def footer(canvas, doc_):
        canvas.saveState()
        w, _ = A4
        canvas.setStrokeColor(gold)
        canvas.setLineWidth(1)
        canvas.line(doc_.leftMargin, 14*mm, w-doc_.rightMargin, 14*mm)
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(grey)
        canvas.drawString(doc_.leftMargin, 9.5*mm, "Strategize — Beyond the Bottom Line")
        canvas.drawRightString(w-doc_.rightMargin, 9.5*mm, f"{L['page']} {canvas.getPageNumber()}")
        canvas.restoreState()

    # Stable header: sizes designed for square PNGs
    legacy_img = _img_contain(legacy_logo_path, max_w_mm=62, max_h_mm=18)
    strat_img  = _img_contain(strategize_logo_path, max_w_mm=48, max_h_mm=18)

    story = []

    # Cover header
    top = Table([[legacy_img or "", strat_img or ""]],
                colWidths=[120*mm, 55*mm],
                rowHeights=[20*mm])
    top.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,0),(0,0),"LEFT"),
        ("ALIGN",(1,0),(1,0),"RIGHT"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(top)
    story.append(Spacer(1, 14))

    story.append(_p("<b>Legacy360°</b>", ParagraphStyle("ct", parent=h1, fontSize=24, leading=28)))
    story.append(_p("Family Governance & Succession Roadmap", ParagraphStyle("cs", parent=h2, fontSize=13, leading=16)))
    story.append(Spacer(1, 6))
    story.append(_p(f"<font color='{gold.hexval()}'>a Strategize service</font>", small))
    story.append(Spacer(1, 14))
    story.append(Table([[""]], colWidths=[175*mm], style=TableStyle([("LINEBELOW",(0,0),(-1,-1),1.2,gold)])))
    story.append(Spacer(1, 12))
    story.append(_p(f"<b>{L['report']}</b>", h2))
    story.append(_p(f"{L['date']}: {datetime.now().strftime('%d/%m/%Y')}", base))
    story.append(PageBreak())

    # Domain Summary table
    story.append(_p(L["summary"], h2))

    dd = df_domains.copy()
    dd["Weight%"] = (dd["weight"]*100).round(0).astype(int)
    dd["Avg"] = dd["avg_score"].round(2)
    dd["Risk"] = dd["risk"].round(3)

    header_row = [L["domain"], L["weight"], L["score"], L["status"], L["risk"]]
    rows = [header_row]

    for _, r in dd.sort_values("risk", ascending=False).iterrows():
        rows.append([
            _p(r["domain"], ParagraphStyle("td", parent=base, fontSize=9, leading=11)),
            _p(f"{int(r['Weight%'])}%", ParagraphStyle("tn", parent=base, fontSize=9, leading=11)),
            _p(f"{r['Avg']:.2f}", ParagraphStyle("tn", parent=base, fontSize=9, leading=11)),
            _p(BAND_LABELS[lang][r["band"]], ParagraphStyle("tn", parent=base, fontSize=9, leading=11)),
            _p(f"{r['Risk']:.3f}", ParagraphStyle("tn", parent=base, fontSize=9, leading=11)),
        ])

    dom_tbl = Table(rows, colWidths=[90*mm, 18*mm, 18*mm, 28*mm, 21*mm], repeatRows=1)
    dom_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),navy),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"DejaVu-Bold"),
        ("FONTSIZE",(0,0),(-1,0),9),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
        ("LEFTPADDING",(0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(dom_tbl)

    story.append(PageBreak())
    story.append(_p(L["appendix"], h2))

    a = answers_df.copy()
    a["domain"] = a["domain_gr"] if lang == "GR" else a["domain_en"]
    a["question"] = a["question_gr"] if lang == "GR" else a["question_en"]

    qa_rows = [["ID", L["domain"], "Question", "Score"]]
    for _, rr in a.iterrows():
        qa_rows.append([
            _p(str(rr["question_id"]), ParagraphStyle("qaid", parent=base, fontSize=8, leading=10)),
            _p(str(rr["domain"]), ParagraphStyle("qad", parent=base, fontSize=8, leading=10)),
            _p(str(rr["question"]), ParagraphStyle("qaq", parent=base, fontSize=8, leading=10)),
            _p(str(rr["score"]), ParagraphStyle("qas", parent=base, fontSize=8, leading=10)),
        ])

    qa_tbl = Table(qa_rows, colWidths=[14*mm, 42*mm, 100*mm, 15*mm], repeatRows=1)
    qa_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),navy),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"DejaVu-Bold"),
        ("FONTSIZE",(0,0),(-1,0),9),
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
        ("LEFTPADDING",(0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(qa_tbl)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


def build_case_pdf(lang: str, case_meta: Dict[str, Any], agg: Dict[str, Any],
                   legacy_logo_path: str, strategize_logo_path: str) -> bytes:
    register_pdf_fonts()

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2C5D")
    gold = colors.HexColor("#C7922B")
    grey = colors.HexColor("#6B7280")

    base = ParagraphStyle("base", parent=styles["BodyText"], fontName="DejaVu", fontSize=10, leading=13)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="DejaVu-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=12, leading=14, textColor=navy, spaceAfter=6)

    title = "Case Αναφορά — Family Alignment" if lang == "GR" else "Case Report — Family Alignment"
    today = datetime.now().strftime("%d/%m/%Y")

    def footer(canvas, doc_):
        canvas.saveState()
        w, _ = A4
        canvas.setStrokeColor(gold)
        canvas.setLineWidth(1)
        canvas.line(doc_.leftMargin, 14*mm, w-doc_.rightMargin, 14*mm)
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(grey)
        canvas.drawString(doc_.leftMargin, 9.5*mm, "Strategize — Beyond the Bottom Line")
        canvas.drawRightString(w-doc_.rightMargin, 9.5*mm, f"{canvas.getPageNumber()}")
        canvas.restoreState()

    # Same stable header logic
    legacy_img = _img_contain(legacy_logo_path, max_w_mm=62, max_h_mm=18)
    strat_img  = _img_contain(strategize_logo_path, max_w_mm=48, max_h_mm=18)

    company = (case_meta.get("company_name") or "").strip()
    case_id = case_meta.get("case_id") or ""

    story = []

    top = Table([[legacy_img or "", strat_img or ""]],
                colWidths=[120*mm, 55*mm],
                rowHeights=[20*mm])
    top.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(0,0),(0,0),"LEFT"),
        ("ALIGN",(1,0),(1,0),"RIGHT"),
        ("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(top)
    story.append(Spacer(1, 14))

    story.append(_p("<b>Legacy360°</b>", ParagraphStyle("ct", parent=h1, fontSize=24, leading=28)))
    story.append(_p(title, ParagraphStyle("cs", parent=h2, fontSize=13, leading=16)))
    story.append(Spacer(1, 12))

    story.append(_p(f"<b>Company:</b> {company or '-'}", base))
    story.append(_p(f"<b>Case ID:</b> {case_id}", base))
    story.append(_p(f"<b>Date:</b> {today}", base))

    story.append(Spacer(1, 10))
    story.append(Table([[""]], colWidths=[175*mm], style=TableStyle([("LINEBELOW",(0,0),(-1,-1),1.0,gold)])))
    story.append(PageBreak())

    overall_avg = agg.get("overall_avg", float("nan"))
    n = agg.get("participants_n", 0)

    story.append(_p("Average Overall Index (0–100)" if lang == "EN" else "Μέσος Συνολικός Δείκτης (0–100)", h2))
    story.append(_p(f"<b>{overall_avg:.1f}</b>", ParagraphStyle("big", parent=h1, fontSize=22, leading=26)))
    story.append(_p(("Participants" if lang == "EN" else "Συμμετέχοντες") + f": <b>{n}</b>", base))
    story.append(Spacer(1, 10))

    rows = [["Domain" if lang == "EN" else "Ενότητα", "Avg" if lang == "EN" else "Μ.Ο.", "Std"]]
    for d in DOMAINS:
        rows.append([
            _p(DOMAIN_LABELS[lang][d.key], ParagraphStyle("d", parent=base, fontSize=9, leading=11)),
            _p(f"{agg['domain_avg'].get(d.key, float('nan')):.2f}", ParagraphStyle("n", parent=base, fontSize=9, leading=11)),
            _p(f"{agg['domain_std'].get(d.key, 0.0):.2f}", ParagraphStyle("n", parent=base, fontSize=9, leading=11)),
        ])

    tbl = Table(rows, colWidths=[125*mm, 22*mm, 22*mm], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),navy),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"DejaVu-Bold"),
        ("FONTSIZE",(0,0),(-1,0),9),
        ("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("ALIGN",(1,1),(-1,-1),"CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke, colors.white]),
        ("LEFTPADDING",(0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    story.append(tbl)

    story.append(Spacer(1, 14))
    story.append(_p("Interpretation & Alignment" if lang == "EN" else "Ερμηνεία & Ευθυγράμμιση", h2))

    # Simple alignment narrative based on Std
    case_df = agg["case_df"].copy()
    high_var = case_df.sort_values("std", ascending=False).head(3)
    low_score = case_df.sort_values("avg_score", ascending=True).head(3)

    if lang == "EN":
        story.append(_p("Higher standard deviation indicates lower alignment across respondents.", base))
        story.append(_p("<b>Top misalignment areas:</b> " + ", ".join(high_var["domain"].tolist()), base))
        story.append(_p("<b>Lowest maturity areas:</b> " + ", ".join(low_score["domain"].tolist()), base))
    else:
        story.append(_p("Υψηλότερη τυπική απόκλιση σημαίνει χαμηλότερη ευθυγράμμιση μεταξύ των συμμετεχόντων.", base))
        story.append(_p("<b>Περιοχές με τη μεγαλύτερη απόκλιση:</b> " + ", ".join(high_var["domain"].tolist()), base))
        story.append(_p("<b>Χαμηλότερες βαθμολογίες ωριμότητας:</b> " + ", ".join(low_score["domain"].tolist()), base))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# =========================================================
# DB RPC (participant) + DB (admin)
# =========================================================

def db_participant_validate_invite(raw_token: str) -> Dict[str, Any]:
    sb = supabase_client(use_service_role=False)
    token_hash = sha256_hex(raw_token)
    res = sb.rpc("validate_invite", {"p_token_hash": token_hash}).execute()
    if not res.data:
        return {"valid": False}
    row = res.data[0]  # case_id, invite_id, participant_email, status
    return {"valid": True, "token_hash": token_hash, **row}

def db_participant_submit(raw_token: str, lang: str, answers_json: Dict[str, int], profile_json: Dict[str, Any], derived_json: Dict[str, Any]) -> Any:
    sb = supabase_client(use_service_role=False)
    token_hash = sha256_hex(raw_token)
    res = sb.rpc("submit_assessment", {
        "p_token_hash": token_hash,
        "p_lang": lang,
        "p_questionnaire_version": QUESTIONNAIRE_VERSION,
        "p_answers": answers_json,
        "p_profile": profile_json,
        "p_derived": derived_json,
    }).execute()
    return res.data

def db_admin_list_cases(limit: int = 200) -> List[Dict[str, Any]]:
    sb = supabase_client(use_service_role=True)
    res = sb.table("cases").select("*").order("created_at", desc=True).limit(limit).execute()
    return res.data or []

def db_admin_create_case(payload: Dict[str, Any]) -> str:
    sb = supabase_client(use_service_role=True)
    res = sb.table("cases").insert(payload).execute()
    return res.data[0]["case_id"]

def db_admin_create_invite(case_id: str, participant_email: str, expires_days: int = 14, max_uses: int = 1) -> Dict[str, str]:
    sb = supabase_client(use_service_role=True)
    raw_token = secrets.token_urlsafe(32)
    token_hash = sha256_hex(raw_token)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()

    ins = {
        "case_id": case_id,
        "participant_email": participant_email,
        "token_hash": token_hash,
        "token_expires_at": expires_at,
        "max_uses": max_uses,
        "uses_count": 0,
        "status": "ACTIVE",
    }
    res = sb.table("invites").insert(ins).execute()
    invite_id = res.data[0]["invite_id"]
    return {"invite_id": invite_id, "raw_token": raw_token}

def db_admin_get_case(case_id: str) -> Dict[str, Any]:
    sb = supabase_client(use_service_role=True)
    res = sb.table("cases").select("*").eq("case_id", case_id).limit(1).execute()
    return res.data[0] if res.data else {}

def db_admin_get_submissions(case_id: str) -> List[Dict[str, Any]]:
    sb = supabase_client(use_service_role=True)
    res = sb.table("submissions").select("*").eq("case_id", case_id).order("submitted_at", desc=True).execute()
    return res.data or []


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Legacy360°", layout="wide")
BASE_DIR = os.path.dirname(__file__)

with st.sidebar.expander("🔧 Runtime Debug (assets/fonts)", expanded=True):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("ROOT:", os.listdir(BASE_DIR))

    assets_dir = os.path.join(BASE_DIR, "assets")
    fonts_dir = os.path.join(assets_dir, "fonts")

    st.write("assets exists:", os.path.exists(assets_dir))
    if os.path.exists(assets_dir):
        st.write("assets listing:", os.listdir(assets_dir))

    st.write("fonts exists:", os.path.exists(fonts_dir))
    if os.path.exists(fonts_dir):
        st.write("fonts listing:", os.listdir(fonts_dir))

    st.write("DejaVuSans.ttf:", os.path.exists(os.path.join(fonts_dir, "DejaVuSans.ttf")))
    st.write("DejaVuSans-Bold.ttf:", os.path.exists(os.path.join(fonts_dir, "DejaVuSans-Bold.ttf")))

import os

BASE_DIR = os.path.dirname(__file__)

st.sidebar.markdown("### 🔧 Runtime Filesystem Debug")
st.sidebar.write("BASE_DIR:", BASE_DIR)
st.sidebar.write("ROOT listing:", os.listdir(BASE_DIR))

assets_dir = os.path.join(BASE_DIR, "assets")
fonts_dir = os.path.join(assets_dir, "fonts")

st.sidebar.write("assets exists:", os.path.exists(assets_dir))
if os.path.exists(assets_dir):
    st.sidebar.write("assets listing:", os.listdir(assets_dir))

st.sidebar.write("fonts exists:", os.path.exists(fonts_dir))
if os.path.exists(fonts_dir):
    st.sidebar.write("fonts listing:", os.listdir(fonts_dir))

st.sidebar.write("DejaVuSans.ttf:", os.path.exists(os.path.join(fonts_dir, "DejaVuSans.ttf")))
st.sidebar.write("DejaVuSans-Bold.ttf:", os.path.exists(os.path.join(fonts_dir, "DejaVuSans-Bold.ttf")))

params = st.query_params
is_admin = str(params.get("admin", "")).strip() in ("1", "true", "yes")
token = str(params.get("token", "")).strip()

lang = st.sidebar.radio("Γλώσσα / Language", ["GR", "EN"], index=0)

BASE_DIR = os.path.dirname(__file__)
st.sidebar.markdown("### 🔧 Font Debug")
st.sidebar.write("BASE_DIR:", BASE_DIR)
st.sidebar.write("assets exists:", os.path.exists(os.path.join(BASE_DIR, "assets")))
st.sidebar.write("fonts dir exists:", os.path.exists(os.path.join(BASE_DIR, "assets", "fonts")))
st.sidebar.write("DejaVuSans.ttf:", os.path.exists(os.path.join(BASE_DIR, "assets", "fonts", "DejaVuSans.ttf")))
st.sidebar.write("DejaVuSans-Bold.ttf:", os.path.exists(os.path.join(BASE_DIR, "assets", "fonts", "DejaVuSans-Bold.ttf")))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LEGACY_LOGO = os.path.join(ASSETS_DIR, "legacy360.png")
STRATEGIZE_LOGO = os.path.join(ASSETS_DIR, "strategize.png")

def header():
    left, right = st.columns([0.68, 0.32], vertical_alignment="center")
    with left:
        if os.path.exists(LEGACY_LOGO):
            st.image(LEGACY_LOGO, width=280)
        st.title(UI[lang]["title"])
        st.caption(UI[lang]["tagline"])
    with right:
        if os.path.exists(STRATEGIZE_LOGO):
            st.image(STRATEGIZE_LOGO, width=240)
    st.markdown("<hr style='border:1px solid #C7922B; margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)


def admin_dashboard():
    header()
    st.subheader("Admin Access")

    admin_pass = _get_secret("ADMIN_PASSWORD", required=True)
    if "admin_ok" not in st.session_state:
        st.session_state["admin_ok"] = False

    if not st.session_state["admin_ok"]:
        pw = st.text_input(UI[lang]["admin_password"], type="password")
        if st.button("Login"):
            if pw == admin_pass:
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error(UI[lang]["admin_wrong"])
        st.stop()

    tabs = st.tabs(["Cases", "Create Case", "Invites", "Aggregation"])

    with tabs[0]:
        cases = db_admin_list_cases()
        st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)

    with tabs[1]:
        company_name = st.text_input("Company name")
        industry = st.text_input("Industry")
        country = st.text_input("Country")
        size_band = st.text_input("Size band (optional)")
        created_by = st.text_input("Created by (optional)")

        if st.button("Create", use_container_width=True, disabled=not company_name.strip()):
            case_id = db_admin_create_case({
                "company_name": company_name.strip(),
                "industry": industry.strip() or None,
                "country": country.strip() or None,
                "size_band": size_band.strip() or None,
                "created_by": created_by.strip() or None,
            })
            st.success(f"Created case_id: {case_id}")

    with tabs[2]:
        case_id = st.text_input("Case ID (uuid)")
        email = st.text_input("Participant email")
        expires_days = st.number_input("Expires in days", min_value=1, max_value=60, value=14)
        max_uses = st.number_input("Max uses", min_value=1, max_value=5, value=1)
        base_url = st.text_input("Participant app base URL (e.g., https://xxx.streamlit.app)", value="")

        if st.button("Generate Invite", use_container_width=True, disabled=not(case_id.strip() and email.strip())):
            inv = db_admin_create_invite(case_id.strip(), email.strip(), int(expires_days), int(max_uses))
            link = f"{base_url.strip().rstrip('/')}/?token={inv['raw_token']}" if base_url.strip() else ""
            st.code(f"Invite ID: {inv['invite_id']}\nToken (raw): {inv['raw_token']}\nLink: {link}".strip())

    with tabs[3]:
        case_id = st.text_input("Case ID to aggregate (uuid)", key="case_id_agg")
        if not case_id.strip():
            st.info("Enter a case_id")
            st.stop()

        case_meta = db_admin_get_case(case_id.strip())
        subs = db_admin_get_submissions(case_id.strip())
        if not subs:
            st.warning("No submissions yet.")
            st.stop()

        agg = aggregate_case(lang, subs)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Avg Overall (0–100)", f"{agg['overall_avg']:.1f}")
        with k2:
            st.metric("Participants", f"{agg['participants_n']}")
        with k3:
            st.metric("Domains", f"{len(DOMAINS)}")

        labels = [DOMAIN_LABELS[lang][d.key] for d in DOMAINS]
        values = [agg["domain_avg"].get(d.key, float("nan")) for d in DOMAINS]
        st.plotly_chart(make_radar(labels, values, "Family Alignment"), use_container_width=True)

        case_df = agg["case_df"].copy()
        case_df["Weight %"] = (case_df["weight"] * 100).round(0).astype(int)
        case_df["Avg (1–5)"] = case_df["avg_score"].round(2)
        case_df["Std"] = case_df["std"].round(2)
        case_df["Band"] = case_df["band"].map(BAND_LABELS[lang])
        st.dataframe(
            case_df[["domain", "Weight %", "Avg (1–5)", "Std", "Band", "risk"]].sort_values("risk", ascending=False),
            use_container_width=True, hide_index=True
        )

        pdf = build_case_pdf(lang, case_meta, agg, LEGACY_LOGO, STRATEGIZE_LOGO)
        st.download_button(
            UI[lang]["download_case_pdf"],
            data=pdf,
            file_name="Legacy360_Case_Alignment.pdf" if lang == "EN" else "Legacy360_Case_Ευθυγράμμιση.pdf",
            mime="application/pdf",
            use_container_width=True
        )


def participant_wizard():
    header()

    if not token:
        st.error(UI[lang]["missing_token"])
        st.stop()

    v = db_participant_validate_invite(token)
    if not v.get("valid"):
        st.error(UI[lang]["token_invalid"])
        st.stop()

    token_status = str(v.get("status") or "").upper()
    read_only = (token_status == "USED")  # after submit: allow view/download, block new submit

    case_id = v.get("case_id")
    participant_email = v.get("participant_email") or ""

    if read_only:
        st.info(UI[lang]["token_used_readonly"])

    st.subheader(UI[lang]["profile"])
    c1, c2, c3 = st.columns(3)
    with c1:
        full_name = st.text_input("Full name (optional)")
        email = st.text_input("Email", value=participant_email, disabled=True)
        role_category = st.selectbox("Role category", ["", "Owner", "Family shareholder", "CEO", "Executive", "Board member", "Next gen", "Other"])
    with c2:
        generation = st.selectbox("Generation", ["", "Gen 1", "Gen 2", "Gen 3", "Gen 4+"])
        age_band = st.selectbox("Age band", ["", "<30", "30–39", "40–49", "50–59", "60+"])
        works_in_business = st.selectbox("Works in business", ["", "Yes", "No"])
    with c3:
        ownership = st.selectbox("Ownership", ["", "Yes", "No"])
        board_member = st.selectbox("Board member", ["", "Yes", "No"])
        st.caption(f"{UI[lang]['case']}: {case_id}")

    profile_json = {
        "full_name": full_name.strip() or None,
        "email": participant_email,
        "role_category": role_category or None,
        "generation": generation or None,
        "age_band": age_band or None,
        "works_in_business": (works_in_business == "Yes") if works_in_business else None,
        "ownership": (ownership == "Yes") if ownership else None,
        "board_member": (board_member == "Yes") if board_member else None,
    }

    st.divider()

    if "answers" not in st.session_state:
        st.session_state["answers"] = {q.id: None for q in QUESTIONS}
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    total_q = len(QUESTIONS)
    answered = sum(1 for vv in st.session_state["answers"].values() if vv is not None)
    ratio = answered / total_q if total_q else 0.0

    st.markdown(f"### {UI[lang]['progress']}")
    st.progress(ratio)
    st.caption(f"{int(round(ratio*100))}% ({answered}/{total_q})")
    st.divider()

    dq = domain_questions_map()

    # Sidebar sections status (restores "left menu" feel)
    with st.sidebar:
        st.markdown("### 🧭 Sections")
        for i, d in enumerate(DOMAINS):
            dom_key = d.key
            missing_dom = [qid for qid in dq[dom_key] if st.session_state["answers"][qid] is None]
            done = (len(missing_dom) == 0)
            marker = "✅" if done else "⬜"
            current = "➡️ " if i == st.session_state["step"] else ""
            st.markdown(f"{current}{marker} {DOMAIN_LABELS[lang][dom_key]}")

    # Wizard pages
    if st.session_state["step"] < len(DOMAINS):
        d = DOMAINS[st.session_state["step"]]
        dom_key = d.key
        st.markdown(f"## 🧭 {DOMAIN_LABELS[lang][dom_key]}")
        st.caption(f"Weight: {int(d.weight*100)}%")

        for q in [qq for qq in QUESTIONS if qq.domain_key == dom_key]:
            options = ["—", 1, 2, 3, 4, 5]
            current = st.session_state["answers"][q.id]
            idx = 0 if current is None else options.index(current)

            val = st.selectbox(
                label=f"**{q.id}** — {q.text[lang]}",
                options=options,
                index=idx,
                key=f"q_{q.id}"
            )
            st.session_state["answers"][q.id] = None if val == "—" else int(val)

        missing = [qid for qid in dq[dom_key] if st.session_state["answers"][qid] is None]
        if missing:
            st.warning(f"{len(missing)} unanswered questions remain in this section.")

        st.divider()
        left, right = st.columns([0.35, 0.65])

        with left:
            if st.session_state["step"] > 0:
                if st.button("⬅️ Previous", use_container_width=True, key=f"btn_prev_{st.session_state['step']}"):
                    st.session_state["step"] = max(st.session_state["step"] - 1, 0)
                    st.rerun()

        with right:
            can_go = (len(missing) == 0)
            is_last = (st.session_state["step"] == len(DOMAINS) - 1)

            if is_last:
                if st.button("Δες τα Αποτελέσματα / See Results 📊",
                             use_container_width=True, disabled=not can_go,
                             key=f"btn_results_{st.session_state['step']}"):
                    st.session_state["step"] = len(DOMAINS)
                    st.rerun()
            else:
                if st.button("Επόμενη Ενότητα / Next Section ➡️",
                             use_container_width=True, disabled=not can_go,
                             key=f"btn_next_{st.session_state['step']}"):
                    st.session_state["step"] = min(st.session_state["step"] + 1, len(DOMAINS))
                    st.rerun()
        return

    # Results page
    st.markdown(f"## 📊 {UI[lang]['results']}")

    if any(vv is None for vv in st.session_state["answers"].values()):
        st.error("Some questions are unanswered. Please complete all sections.")
        st.stop()

    answers_json = {k: int(vv) for k, vv in st.session_state["answers"].items()}
    domain_scores = compute_domain_scores(answers_json)
    overall = weighted_index(domain_scores)
    df = build_domain_df(lang, domain_scores)

    labels = [DOMAIN_LABELS[lang][d.key] for d in DOMAINS]
    values = [domain_scores[d.key] for d in DOMAINS]
    st.plotly_chart(make_radar(labels, values, UI[lang]["results"]), use_container_width=True)

    # Submit lock + safeguard
    if not st.session_state["submitted"]:
        if read_only:
            st.warning(UI[lang]["token_used_readonly"])
        else:
            st.info("Press Submit to store and lock results.")
            if st.button(UI[lang]["submit"], use_container_width=True, disabled=read_only):
                derived_json = {
                    "domain_scores": {k: float(v) for k, v in domain_scores.items()},
                    "overall": float(overall)
                }
                try:
                    db_participant_submit(token, lang, answers_json, profile_json, derived_json)
                    st.session_state["submitted"] = True
                    st.success(UI[lang]["submitted_ok"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Submission failed: {e}")
                    st.stop()
        st.stop()

    # Domain table
    show = df.copy()
    show["Weight %"] = (show["weight"]*100).round(0).astype(int)
    show["Avg (1–5)"] = show["avg_score"].round(2)
    show["Band"] = show["band"].map(BAND_LABELS[lang])
    show["Risk"] = show["risk"].round(3)
    st.dataframe(show[["domain","Weight %","Avg (1–5)","Band","Risk"]], use_container_width=True, hide_index=True)

    # PDF export
    out_rows = []
    for q in QUESTIONS:
        out_rows.append({
            "question_id": q.id,
            "domain_gr": DOMAIN_LABELS["GR"][q.domain_key],
            "domain_en": DOMAIN_LABELS["EN"][q.domain_key],
            "question_gr": q.text["GR"],
            "question_en": q.text["EN"],
            "score": answers_json[q.id],
        })
    out = pd.DataFrame(out_rows)

    pdf = build_participant_pdf(lang, df, float(overall), out, LEGACY_LOGO, STRATEGIZE_LOGO)
    st.download_button(
        UI[lang]["download_pdf"],
        data=pdf,
        file_name="Legacy360_Report.pdf" if lang == "EN" else "Legacy360_Αναφορά.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# =========================================================
# ENTRY
# =========================================================

if is_admin:
    admin_dashboard()
else:
    participant_wizard()




