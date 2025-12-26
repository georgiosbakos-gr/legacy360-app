# legacy360_app.py
# Legacy360° — Family Governance & Succession Roadmap (a Strategize service)
# Streamlit wizard app with validation, progress, submit-lock, premium PDF export + commercial CTA

import os
from io import BytesIO
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import mm


# =========================
# Data model
# =========================

@dataclass
class Domain:
    key: str
    weight: float  # e.g., 0.20


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
    Question("1.1", "corp_gov", {
        "EN": "The roles and responsibilities of the Board, Management, and Shareholders are clearly defined and respected in practice.",
        "GR": "Οι ρόλοι και οι αρμοδιότητες του Διοικητικού Συμβουλίου, της Διοίκησης και των Μετόχων είναι σαφώς καθορισμένοι και γίνονται σεβαστοί στην πράξη."
    }),
    Question("1.2", "corp_gov", {
        "EN": "The Board provides effective strategic oversight and constructive challenge to management decisions.",
        "GR": "Το Διοικητικό Συμβούλιο ασκεί ουσιαστικό στρατηγικό έλεγχο και ασκεί εποικοδομητική κριτική στις αποφάσεις της Διοίκησης."
    }),
    Question("1.3", "corp_gov", {
        "EN": "Decision-making authority and escalation mechanisms are clearly defined and consistently applied.",
        "GR": "Οι αρμοδιότητες λήψης αποφάσεων και οι μηχανισμοί κλιμάκωσης είναι σαφώς καθορισμένοι και εφαρμόζονται με συνέπεια."
    }),
    Question("1.4", "corp_gov", {
        "EN": "Governance structures support accountability, transparency, and long-term value creation.",
        "GR": "Οι δομές διακυβέρνησης υποστηρίζουν τη λογοδοσία, τη διαφάνεια και τη μακροπρόθεσμη δημιουργία αξίας."
    }),

    # Family Governance
    Question("2.1", "family_gov", {
        "EN": "The relationship between the family, ownership, and the business is clearly structured and formally governed.",
        "GR": "Η σχέση μεταξύ Οικογένειας, Ιδιοκτησίας και Επιχείρησης είναι σαφώς δομημένη και διέπεται από τυπικούς κανόνες."
    }),
    Question("2.2", "family_gov", {
        "EN": "There are established forums or processes for family communication, alignment, and conflict resolution.",
        "GR": "Υπάρχουν θεσμοθετημένα όργανα ή διαδικασίες για την επικοινωνία, την ευθυγράμμιση και την επίλυση διαφορών εντός της οικογένειας."
    }),
    Question("2.3", "family_gov", {
        "EN": "Family policies (e.g. employment, dividends, ownership transfers) are clearly defined and applied consistently.",
        "GR": "Οι οικογενειακές πολιτικές (π.χ. απασχόληση, μερίσματα, μεταβίβαση ιδιοκτησίας) είναι σαφώς καθορισμένες και εφαρμόζονται με συνέπεια."
    }),
    Question("2.4", "family_gov", {
        "EN": "Family involvement supports business continuity rather than creating operational or governance risk.",
        "GR": "Η εμπλοκή της οικογένειας υποστηρίζει τη βιωσιμότητα της επιχείρησης και δεν δημιουργεί λειτουργικούς ή διακυβερνητικούς κινδύνους."
    }),

    # Roles of family members
    Question("3.1", "family_roles", {
        "EN": "The roles and responsibilities of family members working in the business are clearly defined and documented.",
        "GR": "Οι ρόλοι και οι αρμοδιότητες των μελών της οικογένειας που εργάζονται στην επιχείρηση είναι σαφώς καθορισμένοι και τεκμηριωμένοι."
    }),
    Question("3.2", "family_roles", {
        "EN": "Entry, progression, and exit criteria for family members are based on objective and transparent principles.",
        "GR": "Τα κριτήρια εισόδου, εξέλιξης και αποχώρησης των μελών της οικογένειας βασίζονται σε αντικειμενικές και διαφανείς αρχές."
    }),
    Question("3.3", "family_roles", {
        "EN": "The performance of family members is evaluated using the same standards applied to non-family executives.",
        "GR": "Η απόδοση των μελών της οικογένειας αξιολογείται με τα ίδια κριτήρια που εφαρμόζονται και στα μη οικογενειακά στελέχη."
    }),
    Question("3.4", "family_roles", {
        "EN": "Family roles within the business add measurable value and do not rely on informal authority.",
        "GR": "Οι ρόλοι των μελών της οικογένειας στην επιχείρηση προσθέτουν μετρήσιμη αξία και δεν βασίζονται σε άτυπη εξουσία."
    }),

    # Strategic Clarity
    Question("4.1", "strategy", {
        "EN": "The organisation has a clearly articulated strategy that is understood across leadership levels.",
        "GR": "Ο οργανισμός διαθέτει σαφώς διατυπωμένη στρατηγική που είναι κατανοητή σε όλα τα επίπεδα ηγεσίας."
    }),
    Question("4.2", "strategy", {
        "EN": "Strategic priorities are translated into clear objectives, initiatives, and execution plans.",
        "GR": "Οι στρατηγικές προτεραιότητες μεταφράζονται σε σαφείς στόχους, πρωτοβουλίες και σχέδια υλοποίησης."
    }),
    Question("4.3", "strategy", {
        "EN": "Strategic decision-making reflects agreed priorities rather than short-term or ad-hoc considerations.",
        "GR": "Η λήψη στρατηγικών αποφάσεων αντανακλά συμφωνημένες προτεραιότητες και όχι βραχυπρόθεσμες ή αποσπασματικές επιλογές."
    }),
    Question("4.4", "strategy", {
        "EN": "The strategy balances business performance with family expectations and long-term continuity.",
        "GR": "Η στρατηγική εξισορροπεί την επιχειρησιακή απόδοση με τις προσδοκίες της οικογένειας και τη μακροπρόθεσμη συνέχεια."
    }),

    # Financial & performance visibility
    Question("5.1", "fin_perf", {
        "EN": "Financial and performance information is timely, reliable, and decision-relevant.",
        "GR": "Η χρηματοοικονομική και επιχειρησιακή πληροφόρηση είναι έγκαιρη, αξιόπιστη και κατάλληλη για τη λήψη αποφάσεων."
    }),
    Question("5.2", "fin_perf", {
        "EN": "Key performance indicators (KPIs) are clearly defined and aligned with strategic priorities.",
        "GR": "Οι βασικοί δείκτες απόδοσης (KPIs) είναι σαφώς καθορισμένοι και ευθυγραμμισμένοι με τις στρατηγικές προτεραιότητες."
    }),
    Question("5.3", "fin_perf", {
        "EN": "Performance discussions focus on insight and forward-looking actions, not only historical results.",
        "GR": "Οι συζητήσεις απόδοσης εστιάζουν σε ουσιαστική ανάλυση και μελλοντικές ενέργειες, και όχι μόνο σε ιστορικά αποτελέσματα."
    }),
    Question("5.4", "fin_perf", {
        "EN": "Transparency supports accountability at both management and ownership levels.",
        "GR": "Η διαφάνεια υποστηρίζει τη λογοδοσία τόσο σε επίπεδο Διοίκησης όσο και Ιδιοκτησίας."
    }),

    # Sustainability & continuity
    Question("6.1", "sust_cont", {
        "EN": "There is a clear and realistic succession approach for key leadership and ownership roles.",
        "GR": "Υπάρχει σαφής και ρεαλιστική προσέγγιση διαδοχής για κρίσιμους ρόλους ηγεσίας και ιδιοκτησίας."
    }),
    Question("6.2", "sust_cont", {
        "EN": "The organisation actively manages risks that could affect long-term business and family continuity.",
        "GR": "Ο οργανισμός διαχειρίζεται ενεργά τους κινδύνους που θα μπορούσαν να επηρεάσουν τη μακροπρόθεσμη συνέχεια της επιχείρησης και της οικογένειας."
    }),
    Question("6.3", "sust_cont", {
        "EN": "Leadership development and talent pipelines support future organisational needs.",
        "GR": "Η ανάπτυξη ηγεσίας και η δεξαμενή ταλέντων υποστηρίζουν τις μελλοντικές ανάγκες του οργανισμού."
    }),
    Question("6.4", "sust_cont", {
        "EN": "Sustainability considerations are integrated into strategic and governance decision-making.",
        "GR": "Οι παράμετροι βιωσιμότητας ενσωματώνονται στη στρατηγική και στη λήψη αποφάσεων διακυβέρνησης."
    }),
]


# =========================
# UI strings
# =========================

UI = {
    "GR": {
        "app_title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "intro_title": "Αυτοαξιολόγηση (Self-completed)",
        "intro_body": (
            "Συμπληρώστε την αξιολόγηση με βάση την πραγματική κατάσταση. "
            "Στο τέλος θα δείτε dashboard, προτεραιότητες και δυνατότητα εξαγωγής PDF/CSV."
        ),
        "scale_title": "Κλίμακα Ωριμότητας 1–5 (Ορισμοί)",
        "scale": {
            1: "Άτυπο / Αποσπασματικό: εξάρτηση από πρόσωπα, χωρίς σταθερή δομή ή τεκμηρίωση.",
            2: "Μερικώς Ορισμένο: υπάρχουν πρακτικές αλλά ασυνέπεια/επιλεκτική εφαρμογή.",
            3: "Ορισμένο αλλά όχι πλήρως ενσωματωμένο: δομές υπάρχουν, η εφαρμογή δεν είναι σταθερή.",
            4: "Ενσωματωμένο & αποτελεσματικό: σαφές, συνεπές, υποστηρίζει ποιοτικές αποφάσεις.",
            5: "Προηγμένο / Πρότυπο: πλήρως ενσωματωμένο, με συστηματική αναθεώρηση και υψηλή ωριμότητα."
        },
        "question_help": "Επιλέξτε βαθμό 1–5 για να συνεχίσετε.",
        "results": "Αποτελέσματα",
        "overall_index": "Συνολικός Δείκτης Ωριμότητας (0–100)",
        "priority_title": "Κορυφαίες Προτεραιότητες (Top Focus Areas)",
        "download_csv": "Λήψη CSV",
        "download_pdf": "Λήψη PDF",
        "incomplete_all": "Υπάρχουν ερωτήσεις χωρίς απάντηση. Παρακαλώ συμπληρώστε όλες τις ερωτήσεις.",
        "interpretations": "Ερμηνεία & Επιπτώσεις Συζήτησης",
        "overall_interp_title": "Συνοπτική Ερμηνεία",
        "risk_matrix": "Χάρτης Συγκέντρωσης Κινδύνου (Score × Weight)",
        "radar": "Radar Ωριμότητας",
        "bars": "Ανά Ενότητα (Μέσος Όρος 1–5)",
        "submit_info": "Πατήστε Υποβολή για να κλειδώσετε τα αποτελέσματα.",
        "submit_btn": "✅ Υποβολή / Submit",
        "back_btn": "⬅️ Προηγούμενο / Previous",
        "next_btn": "Επόμενη Ενότητα / Next Section ➡️",
        "see_results_btn": "Δες τα Αποτελέσματα / See Results 📊",
        "missing_count": "Απομένουν {n} ερωτήσεις χωρίς απάντηση σε αυτή την ενότητα.",
        "cta_expander_title": "Σύνοψη Επόμενων Ενεργειών & Προτάσεων",
    },
    "EN": {
        "app_title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "intro_title": "Self-completed assessment",
        "intro_body": (
            "Complete the assessment based on current reality. "
            "At the end you will receive a dashboard with priorities and PDF/CSV export."
        ),
        "scale_title": "Maturity Scale 1–5 (Anchors)",
        "scale": {
            1: "Informal / ad-hoc: person-dependent, no consistent structure or documentation.",
            2: "Partially defined: some practices exist but inconsistent / selectively applied.",
            3: "Defined but not embedded: structures exist; adoption and compliance vary.",
            4: "Embedded & effective: clearly defined and consistently applied; supports decision quality.",
            5: "Advanced / role model: fully embedded, continuously reviewed; maturity beyond peers."
        },
        "question_help": "Select a score 1–5 to continue.",
        "results": "Results",
        "overall_index": "Overall Maturity Index (0–100)",
        "priority_title": "Top Focus Areas",
        "download_csv": "Download CSV",
        "download_pdf": "Download PDF",
        "incomplete_all": "Some questions are unanswered. Please complete all questions.",
        "interpretations": "Interpretation & Discussion Implications",
        "overall_interp_title": "Executive Summary Interpretation",
        "risk_matrix": "Risk Concentration Map (Score × Weight)",
        "radar": "Maturity Radar",
        "bars": "By Domain (Average 1–5)",
        "submit_info": "Press Submit to lock results.",
        "submit_btn": "✅ Submit",
        "back_btn": "⬅️ Previous",
        "next_btn": "Next Section ➡️",
        "see_results_btn": "See Results 📊",
        "missing_count": "{n} questions remain unanswered in this section.",
        "cta_expander_title": "Next Actions & Recommendations",
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

DOMAIN_INTERP = {
    "GR": {
        "RED": "Υπάρχουν ουσιαστικά κενά δομής/εφαρμογής. Ο κίνδυνος κλιμάκωσης (σύγκρουση, καθυστερήσεις, ασυνέπεια αποφάσεων) είναι αυξημένος.",
        "AMBER": "Το πλαίσιο είναι μερικώς ορισμένο αλλά όχι πλήρως ενσωματωμένο. Απαιτείται τυποποίηση, σαφήνεια ρόλων/κανόνων και πειθαρχία εφαρμογής.",
        "GREEN": "Η πρακτική είναι ενσωματωμένη και λειτουργεί αποτελεσματικά. Προτείνεται συστηματική αναθεώρηση και ενίσχυση όπου χρειάζεται.",
    },
    "EN": {
        "RED": "Material structural and adoption gaps exist. Escalation risk (conflict, delays, inconsistent decisions) is elevated.",
        "AMBER": "The framework is partially defined but not fully embedded. Standardisation, role clarity and disciplined application are required.",
        "GREEN": "Practices are embedded and effective. Maintain with periodic review and targeted enhancements.",
    }
}

OVERALL_INTERP = {
    "GR": {
        "RED": "Το συνολικό προφίλ ωριμότητας υποδηλώνει υψηλό διακυβερνητικό και εκτελεστικό κίνδυνο. Συνιστάται άμεση εστίαση στα κρίσιμα πεδία πριν από μεγάλες δεσμεύσεις (επενδύσεις, διαδοχή, ανάπτυξη).",
        "AMBER": "Υπάρχει λειτουργική βάση, αλλά η ωριμότητα δεν είναι ακόμη συστηματικά ενσωματωμένη. Με στοχευμένες παρεμβάσεις σε υψηλού βάρους ενότητες, μειώνεται σημαντικά ο κίνδυνος και ενισχύεται η συνέχεια.",
        "GREEN": "Το προφίλ δείχνει ισχυρή ωριμότητα. Προτεραιότητα: διατήρηση πειθαρχίας, περιοδικές αναθεωρήσεις και προληπτική προετοιμασία διαδοχής/συνέχειας.",
    },
    "EN": {
        "RED": "The overall maturity profile indicates elevated governance and execution risk. Prioritise critical areas before major commitments (investments, succession moves, expansion).",
        "AMBER": "A functional base exists, but maturity is not yet consistently embedded. Targeted interventions in high-weight domains can materially reduce risk and strengthen continuity.",
        "GREEN": "The profile indicates strong maturity. Maintain discipline, run periodic reviews and proactively prepare succession/continuity.",
    }
}

# Commercial bridge (used in App + PDF)
NEXT_ACTIONS = {
    "GR": {
        "title": "Σύνοψη Επόμενων Ενεργειών & Προτάσεων",
        "intro": (
            "Με βάση τα αποτελέσματα, προτείνονται τα ακόλουθα βήματα ώστε η επιχείρηση να μετατρέψει "
            "τη διάγνωση σε στοχευμένο σχέδιο βελτίωσης."
        ),
        "bullets": [
            "Επιβεβαίωση ευρημάτων: workshop 60–90’ με βασικούς decision makers (Owner(s), CEO, Board/Advisors).",
            "Οριστικοποίηση προτεραιοτήτων: επιλογή 3–5 παρεμβάσεων υψηλής αξίας (high-weight domains / υψηλός κίνδυνος).",
            "Ορισμός governance framework: ρόλοι, αρμοδιότητες, escalation, decision rights και cadence συναντήσεων.",
            "Διαδοχή & συνέχεια: επόμενα βήματα για succession readiness, risk controls και talent pipeline.",
            "Μετρήσιμη εφαρμογή: KPIs, milestones και μηχανισμός παρακολούθησης προόδου 8–12 εβδομάδων."
        ],
        "cta_title": "Πώς μπορεί να βοηθήσει η Strategize",
        "cta_body": (
            "Η Strategize διαθέτει σημαντική εμπειρία σε έργα οικογενειακών επιχειρήσεων, "
            "διακυβέρνησης, στρατηγικής σαφήνειας και μετασχηματισμού. Μπορούμε να μετατρέψουμε τα αποτελέσματα "
            "σε decision-grade roadmap με σαφή deliverables, χρονοδιάγραμμα και μηχανισμό εφαρμογής."
        ),
        "cta_button": "Ζητήστε σύντομη συζήτηση 20’",
        "cta_email": "gbakos@strategize.gr",
        "cta_site": "https://strategize.gr",
        "cta_note": "Απάντηση με το PDF ή ένα screenshot των αποτελεσμάτων αρκεί για να ξεκινήσουμε.",
    },
    "EN": {
        "title": "Next Actions & Recommendations",
        "intro": (
            "Based on the results, the following steps are recommended to convert diagnosis into a focused improvement plan."
        ),
        "bullets": [
            "Validate findings: a 60–90’ workshop with key decision makers (Owner(s), CEO, Board/Advisors).",
            "Confirm priorities: select 3–5 high-value interventions (high-weight domains / highest risk).",
            "Define governance framework: roles, decision rights, escalation, meeting cadence and accountability.",
            "Succession & continuity: next steps for succession readiness, risk controls and talent pipeline.",
            "Execution with metrics: KPIs, milestones and an 8–12 week progress drumbeat."
        ],
        "cta_title": "How Strategize can help",
        "cta_body": (
            "Strategize has extensive experience in family business governance, board effectiveness, strategy clarity and transformation execution. "
            "We can translate the assessment into a decision-grade roadmap with clear deliverables, timeline and execution governance."
        ),
        "cta_button": "Request a 20’ review call",
        "cta_email": "gbakos@strategize.gr",
        "cta_site": "https://strategize.gr",
        "cta_note": "Reply with the PDF report (or a screenshot of results) and we can start.",
    }
}


# =========================
# Scoring & charts
# =========================

def band_for_score(score: float) -> str:
    for b, lo, hi in BANDS:
        if lo <= score < hi:
            return b
    return "AMBER"


def weighted_index(domain_scores: Dict[str, float]) -> float:
    total = 0.0
    for d in DOMAINS:
        s = domain_scores.get(d.key, np.nan)
        if np.isnan(s):
            return np.nan
        total += s * d.weight
    # Convert 1–5 to 0–100: 1 => 0, 5 => 100
    return (total - 1.0) / 4.0 * 100.0


def risk_priority(avg_score: float, weight: float) -> float:
    # Higher risk when score is low and weight is high
    return (6.0 - avg_score) * weight


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


# =========================
# PDF export (Premium cover + footer + CTA)
# =========================

def build_pdf_report(
    lang: str,
    df_domains: pd.DataFrame,
    overall_0_100: float,
    overall_band: str,
    answers_df: pd.DataFrame,
    legacy_logo_path: str,
    strategize_logo_path: str,
) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()
    navy = colors.HexColor("#0B2C5D")
    gold = colors.HexColor("#C7922B")
    grey = colors.HexColor("#6B7280")

    base = ParagraphStyle("base", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12, textColor=grey)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, textColor=navy, spaceAfter=6)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=13, textColor=navy, spaceAfter=4)

    L = {
        "GR": {
            "cover_title": "Legacy360°",
            "cover_subtitle": "Family Governance & Succession Roadmap",
            "cover_tagline": "a Strategize service",
            "report_title": "Αναφορά Αποτελεσμάτων",
            "date": "Ημερομηνία",
            "confidential": "ΕΜΠΙΣΤΕΥΤΙΚΟ / CONFIDENTIAL",
            "overall": "Συνολικός Δείκτης Ωριμότητας (0–100)",
            "summary": "Σύνοψη ανά Ενότητα",
            "domain": "Ενότητα",
            "weight": "Βάρος",
            "score": "Βαθμός (1–5)",
            "status": "Κατάσταση",
            "risk": "Κίνδυνος",
            "priorities": "Κορυφαίες Προτεραιότητες (Top Focus Areas)",
            "appendix": "Παράρτημα: Απαντήσεις",
            "question": "Ερώτηση",
            "page": "Σελίδα",
            "next_actions": "Σύνοψη Επόμενων Ενεργειών & Προτάσεων",
            "how_help": "Πώς μπορεί να βοηθήσει η Strategize",
            "contact": "Επικοινωνία",
        },
        "EN": {
            "cover_title": "Legacy360°",
            "cover_subtitle": "Family Governance & Succession Roadmap",
            "cover_tagline": "a Strategize service",
            "report_title": "Results Report",
            "date": "Date",
            "confidential": "CONFIDENTIAL",
            "overall": "Overall Maturity Index (0–100)",
            "summary": "Domain Summary",
            "domain": "Domain",
            "weight": "Weight",
            "score": "Score (1–5)",
            "status": "Status",
            "risk": "Risk",
            "priorities": "Top Focus Areas",
            "appendix": "Appendix: Responses",
            "question": "Question",
            "page": "Page",
            "next_actions": "Next Actions & Recommendations",
            "how_help": "How Strategize can help",
            "contact": "Contact",
        }
    }[lang]

    today = datetime.now().strftime("%d/%m/%Y")

    def try_image(path: str, width_mm: float):
        try:
            if path and os.path.exists(path):
                img = Image(path, width=width_mm * mm, height=width_mm * mm * 0.38)
                return img
        except Exception:
            pass
        return None

    legacy_img = try_image(legacy_logo_path, 65)
    strat_img = try_image(strategize_logo_path, 58)

    def _footer(canvas, doc_):
        canvas.saveState()
        w, _h = A4
        canvas.setStrokeColor(gold)
        canvas.setLineWidth(1)
        canvas.line(doc_.leftMargin, 14 * mm, w - doc_.rightMargin, 14 * mm)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(grey)
        canvas.drawString(doc_.leftMargin, 9.5 * mm, "Strategize — Beyond the Bottom Line")
        canvas.drawRightString(w - doc_.rightMargin, 9.5 * mm, f"{L['page']} {canvas.getPageNumber()}")
        canvas.restoreState()

    story: List = []

    # -------------------------
    # COVER PAGE
    # -------------------------
    top_tbl = Table([[legacy_img if legacy_img else "", strat_img if strat_img else ""]],
                    colWidths=[120 * mm, 55 * mm])
    top_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(top_tbl)
    story.append(Spacer(1, 18))

    story.append(Paragraph(L["cover_title"], ParagraphStyle("coverTitle", parent=h1, fontSize=26, leading=30)))
    story.append(Paragraph(L["cover_subtitle"], ParagraphStyle("coverSub", parent=h2, fontSize=14, leading=18, textColor=navy)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<font color='{gold.hexval()}'>{L['cover_tagline']}</font>", small))
    story.append(Spacer(1, 18))

    story.append(Table([[""]], colWidths=[175 * mm], style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1.2, gold)])))
    story.append(Spacer(1, 18))

    meta_tbl = Table(
        [[Paragraph(f"<b>{L['report_title']}</b>", h2), ""],
         [Paragraph(f"{L['date']}: {today}", base),
          Paragraph(L["confidential"], ParagraphStyle("conf", parent=base, textColor=gold, fontName="Helvetica-Bold"))]],
        colWidths=[120 * mm, 55 * mm]
    )
    meta_tbl.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (1, 1), (1, 1), "RIGHT"),
    ]))
    story.append(meta_tbl)

    story.append(Spacer(1, 240))
    story.append(Paragraph("© Strategize", small))
    story.append(PageBreak())

    # -------------------------
    # MAIN CONTENT
    # -------------------------
    story.append(Paragraph(L["overall"], h2))
    story.append(Paragraph(
        f"<font color='{navy.hexval()}'><b>{overall_0_100:.1f}</b></font>",
        ParagraphStyle("bigNumber", parent=h1, fontSize=22, leading=26)
    ))
    story.append(Paragraph(OVERALL_INTERP[lang][overall_band], base))
    story.append(Spacer(1, 12))

    # Next actions + Strategize CTA (in PDF)
    na = NEXT_ACTIONS[lang]
    story.append(Paragraph(L["next_actions"], h2))
    story.append(Paragraph(na["intro"], base))
    story.append(Spacer(1, 6))
    for b in na["bullets"]:
        story.append(Paragraph(f"• {b}", base))
    story.append(Spacer(1, 8))
    story.append(Paragraph(L["how_help"], h3))
    story.append(Paragraph(na["cta_body"], base))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>{L['contact']}:</b> {na['cta_email']}  |  {na['cta_site']}", base))
    story.append(Paragraph(na["cta_note"], small))
    story.append(Spacer(1, 14))

    # Domain summary table
    story.append(Paragraph(L["summary"], h2))

    dd = df_domains.copy()
    dd["Weight%"] = (dd["weight"] * 100).round(0).astype(int)
    dd["Avg"] = dd["avg_score"].round(2)
    dd["Risk"] = dd["risk"].round(3)

    table_data = [[L["domain"], L["weight"], L["score"], L["status"], L["risk"]]]
    for _, r in dd.sort_values("risk", ascending=False).iterrows():
        table_data.append([
            r["domain"],
            f"{int(r['Weight%'])}%",
            f"{r['Avg']:.2f}",
            BAND_LABELS[lang][r["band"]],
            f"{r['Risk']:.3f}",
        ])

    dom_tbl = Table(table_data, colWidths=[78 * mm, 18 * mm, 22 * mm, 30 * mm, 22 * mm])
    dom_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(dom_tbl)
    story.append(Spacer(1, 12))

    # Priorities
    story.append(Paragraph(L["priorities"], h2))
    top5 = dd.sort_values("risk", ascending=False).head(5)
    for i, r in enumerate(top5.to_dict(orient="records"), start=1):
        story.append(Paragraph(
            f"<b>{i}. {r['domain']}</b> — {L['score']}: {r['Avg']:.2f} · {L['weight']}: {int(r['Weight%'])}% · {L['status']}: {BAND_LABELS[lang][r['band']]}",
            base
        ))
        story.append(Paragraph(DOMAIN_INTERP[lang][r["band"]], small))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # Appendix
    story.append(Paragraph(L["appendix"], h2))

    a = answers_df.copy()
    a["domain"] = a["domain_gr"] if lang == "GR" else a["domain_en"]
    a["question"] = a["question_gr"] if lang == "GR" else a["question_en"]

    qa_data = [["ID", L["domain"], L["question"], L["score"]]]
    for _, rr in a.iterrows():
        qa_data.append([rr["question_id"], rr["domain"], rr["question"], str(rr["score"])])

    qa_tbl = Table(qa_data, colWidths=[12 * mm, 40 * mm, 105 * mm, 15 * mm], repeatRows=1)
    qa_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (-1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(qa_tbl)

    # Apply footer on all pages after cover (i.e., from page 2 onwards)
    def on_first_page(canvas, doc_):
        _footer(canvas, doc_)

    def on_later_pages(canvas, doc_):
        _footer(canvas, doc_)

    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_later_pages)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# =========================
# App setup
# =========================

st.set_page_config(page_title="Legacy360°", layout="wide")

# Language selector (Greek default)
lang = st.sidebar.radio("Language / Γλώσσα", ["GR", "EN"], index=0)

# Assets paths (Cloud-safe)
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LEGACY_LOGO = os.path.join(ASSETS_DIR, "legacy360.png")
STRATEGIZE_LOGO = os.path.join(ASSETS_DIR, "strategize.png")


# Header with logos
header_left, header_right = st.columns([0.68, 0.32], vertical_alignment="center")

with header_left:
    if os.path.exists(LEGACY_LOGO):
        st.image(LEGACY_LOGO, width=280)
    else:
        st.warning("Legacy360 logo not found in assets/ (legacy360.png)")
    st.title(UI[lang]["app_title"])
    st.caption(UI[lang]["tagline"])

with header_right:
    if os.path.exists(STRATEGIZE_LOGO):
        st.image(STRATEGIZE_LOGO, width=240)
    else:
        st.warning("Strategize logo not found in assets/ (strategize.png)")

st.markdown("<hr style='border:1px solid #C7922B; margin-top:10px; margin-bottom:10px;'>", unsafe_allow_html=True)

# Intro + scale
colA, colB = st.columns([0.55, 0.45])
with colA:
    st.subheader(UI[lang]["intro_title"])
    st.write(UI[lang]["intro_body"])
with colB:
    with st.expander(UI[lang]["scale_title"], expanded=True):
        for k in range(1, 6):
            st.markdown(f"**{k}** — {UI[lang]['scale'][k]}")

st.divider()

# Group questions per domain
domain_questions: Dict[str, List[Question]] = {d.key: [] for d in DOMAINS}
for q in QUESTIONS:
    domain_questions[q.domain_key].append(q)

TOTAL_QUESTIONS = len(QUESTIONS)

# Session state
if "answers" not in st.session_state:
    # None means unanswered (no preselection)
    st.session_state["answers"] = {q.id: None for q in QUESTIONS}

if "step" not in st.session_state:
    # 0..len(DOMAINS)-1 domain steps, len(DOMAINS)=results
    st.session_state["step"] = 0

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False


# Helpers
def answered_count() -> int:
    return sum(1 for v in st.session_state["answers"].values() if v is not None)


def completion_ratio() -> float:
    return answered_count() / TOTAL_QUESTIONS


def domain_question_ids(domain_key: str) -> List[str]:
    return [q.id for q in domain_questions[domain_key]]


def domain_is_complete(domain_key: str) -> bool:
    return all(st.session_state["answers"][qid] is not None for qid in domain_question_ids(domain_key))


def go_next():
    st.session_state["step"] = min(st.session_state["step"] + 1, len(DOMAINS))
    st.rerun()


def go_prev():
    st.session_state["step"] = max(st.session_state["step"] - 1, 0)
    st.rerun()


def go_results():
    st.session_state["step"] = len(DOMAINS)
    st.rerun()


# Progress UI
st.markdown("### Progress / Πρόοδος")
pct = int(round(completion_ratio() * 100))
st.progress(completion_ratio())
st.caption(f"{pct}% ({answered_count()}/{TOTAL_QUESTIONS})")

st.divider()

# Sidebar navigation (optional)
with st.sidebar:
    st.markdown("### Navigation / Πλοήγηση")
    nav_labels = [f"{i+1}. {DOMAIN_LABELS[lang][d.key]}" for i, d in enumerate(DOMAINS)] + ["📊 Results / Αποτελέσματα"]
    sel = st.radio(
        " ",
        options=list(range(len(DOMAINS) + 1)),
        format_func=lambda i: nav_labels[i],
        index=st.session_state["step"],
        key="nav_radio"
    )
    if sel != st.session_state["step"]:
        st.session_state["step"] = sel
        st.rerun()


# =========================
# DOMAIN PAGES (wizard)
# =========================
if st.session_state["step"] < len(DOMAINS):
    d = DOMAINS[st.session_state["step"]]
    dom_key = d.key

    st.markdown(f"## 🧭 {DOMAIN_LABELS[lang][dom_key]}")
    st.caption(f"Weight / Βάρος: **{int(d.weight * 100)}%**")
    st.write("")

    for q in domain_questions[dom_key]:
        key = f"ans_{q.id}"
        options = ["—"] + [1, 2, 3, 4, 5]
        current = st.session_state["answers"][q.id]
        idx = 0 if current is None else options.index(current)

        choice = st.selectbox(
            label=f"**{q.id}** — {q.text[lang]}",
            options=options,
            index=idx,
            help=UI[lang]["question_help"],
            key=key
        )

        st.session_state["answers"][q.id] = None if choice == "—" else int(choice)
        st.write("")

    missing_in_domain = [qid for qid in domain_question_ids(dom_key) if st.session_state["answers"][qid] is None]
    if missing_in_domain:
        st.warning(UI[lang]["missing_count"].format(n=len(missing_in_domain)))

    st.divider()

    left_btn, right_btn = st.columns([0.35, 0.65])

    with left_btn:
        if st.session_state["step"] > 0:
            st.button(UI[lang]["back_btn"], use_container_width=True, on_click=go_prev)

    with right_btn:
        is_last_domain = (st.session_state["step"] == len(DOMAINS) - 1)
        can_proceed = domain_is_complete(dom_key)

        if not is_last_domain:
            st.button(
                UI[lang]["next_btn"],
                use_container_width=True,
                disabled=not can_proceed,
                on_click=go_next
            )
        else:
            st.button(
                UI[lang]["see_results_btn"],
                use_container_width=True,
                disabled=not can_proceed,
                on_click=go_results
            )


# =========================
# RESULTS PAGE
# =========================
else:
    st.markdown(f"## 📊 {UI[lang]['results']}")

    # Global validation
    if answered_count() < TOTAL_QUESTIONS:
        st.error(UI[lang]["incomplete_all"])
        st.button(UI[lang]["back_btn"], on_click=go_prev)
        st.stop()

    # Submit lock
    if not st.session_state["submitted"]:
        st.info(UI[lang]["submit_info"])
        if st.button(UI[lang]["submit_btn"], use_container_width=True):
            st.session_state["submitted"] = True
            st.rerun()
        st.stop()

    # Compute domain scores
    domain_scores: Dict[str, float] = {}
    rows = []
    for dd in DOMAINS:
        vals = [st.session_state["answers"][q.id] for q in domain_questions[dd.key]]
        avg = float(np.mean(vals))
        domain_scores[dd.key] = avg
        rows.append({
            "domain_key": dd.key,
            "domain": DOMAIN_LABELS[lang][dd.key],
            "weight": dd.weight,
            "avg_score": avg,
            "band": band_for_score(avg),
            "risk": risk_priority(avg, dd.weight),
        })

    df = pd.DataFrame(rows).sort_values("risk", ascending=False)
    overall = weighted_index(domain_scores)

    # KPI row
    k1, k2, k3 = st.columns([0.34, 0.33, 0.33])
    with k1:
        st.metric(UI[lang]["overall_index"], f"{overall:.1f}")
        st.progress(min(max(overall / 100.0, 0.0), 1.0))
    with k2:
        red_count = int((df["band"] == "RED").sum())
        amber_count = int((df["band"] == "AMBER").sum())
        green_count = int((df["band"] == "GREEN").sum())
        st.metric("Domains (R / A / G)", f"{red_count} / {amber_count} / {green_count}")
    with k3:
        top = df.iloc[0]
        st.metric("Top Risk Domain" if lang == "EN" else "Κορυφαίος Κίνδυνος", f"{top['domain']}")

    st.divider()

    # Charts
    c1, c2 = st.columns([0.52, 0.48])
    labels = [DOMAIN_LABELS[lang][d.key] for d in DOMAINS]
    values = [domain_scores[d.key] for d in DOMAINS]

    with c1:
        st.plotly_chart(make_radar(labels, values, UI[lang]["radar"]), use_container_width=True)

    with c2:
        bar_df = pd.DataFrame({"Domain": labels, "Avg (1–5)": values})
        fig = go.Figure(go.Bar(x=bar_df["Domain"], y=bar_df["Avg (1–5)"]))
        fig.update_layout(
            title=UI[lang]["bars"],
            height=380,
            margin=dict(l=30, r=30, t=50, b=80),
            xaxis_tickangle=-25,
            yaxis=dict(range=[1, 5]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Risk table
    st.subheader(UI[lang]["risk_matrix"])
    show = df.copy()
    show["Weight %"] = (show["weight"] * 100).round(0).astype(int)
    show["Avg (1–5)"] = show["avg_score"].round(2)
    show["Band"] = show["band"].map(BAND_LABELS[lang])
    show["Risk Score"] = show["risk"].round(3)
    show = show[["domain", "Weight %", "Avg (1–5)", "Band", "Risk Score"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

    # Priorities
    st.subheader(UI[lang]["priority_title"])
    pri = df.head(5)
    for _, r in pri.iterrows():
        band = r["band"]
        st.markdown(f"### {'🔴' if band == 'RED' else '🟡' if band == 'AMBER' else '🟢'} {r['domain']}")
        st.caption(f"Weight / Βάρος: {int(r['weight'] * 100)}% · Avg: {r['avg_score']:.2f} · {BAND_LABELS[lang][band]}")
        st.write(DOMAIN_INTERP[lang][band])

    st.divider()

    # Interpretations
    st.subheader(UI[lang]["interpretations"])
    overall_band = band_for_score(float(np.mean(list(domain_scores.values()))))
    st.markdown(f"### {UI[lang]['overall_interp_title']}")
    st.write(OVERALL_INTERP[lang][overall_band])

    # ---------------------------
    # Commercial bridge (ONLY on Results)
    # ---------------------------
    na = NEXT_ACTIONS[lang]
    with st.expander("🚀 " + UI[lang]["cta_expander_title"], expanded=True):
        st.write(na["intro"])
        for b in na["bullets"]:
            st.markdown(f"- {b}")

        st.markdown(f"**{na['cta_title']}**")
        st.write(na["cta_body"])

        cta1, cta2 = st.columns([0.55, 0.45])
        with cta1:
            mailto = (
                f"mailto:{na['cta_email']}?subject=Legacy360%20Results%20Review"
                f"&body=Hello%2C%0A%0AI%20would%20like%20a%2020%E2%80%99%20review%20call%20to%20discuss%20our%20Legacy360%20results.%0A%0ARegards%2C"
            )
            st.markdown(
                f"""
                <a href="{mailto}" target="_blank" style="
                    display:inline-block; padding:12px 16px; border-radius:10px;
                    background:#0B2C5D; color:white; text-decoration:none; font-weight:600;">
                    {na['cta_button']}
                </a>
                """,
                unsafe_allow_html=True
            )
        with cta2:
            st.markdown(f"**Email:** {na['cta_email']}")
            st.markdown(f"**Website:** {na['cta_site']}")
            st.caption(na["cta_note"])

    st.divider()

    # Build answers dataframe for export
    out_rows = []
    for q in QUESTIONS:
        out_rows.append({
            "question_id": q.id,
            "domain_key": q.domain_key,
            "domain_gr": DOMAIN_LABELS["GR"][q.domain_key],
            "domain_en": DOMAIN_LABELS["EN"][q.domain_key],
            "question_gr": q.text["GR"],
            "question_en": q.text["EN"],
            "score": st.session_state["answers"][q.id],
        })
    out = pd.DataFrame(out_rows)

    # CSV download
    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        UI[lang]["download_csv"],
        data=csv_bytes,
        file_name="legacy360_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    # PDF download
    pdf_bytes = build_pdf_report(
        lang=lang,
        df_domains=df,
        overall_0_100=overall,
        overall_band=overall_band,
        answers_df=out,
        legacy_logo_path=LEGACY_LOGO,
        strategize_logo_path=STRATEGIZE_LOGO,
    )

    pdf_filename = "Legacy360_Report.pdf" if lang == "EN" else "Legacy360_Αναφορά.pdf"
    st.download_button(
        UI[lang]["download_pdf"],
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True
    )

    st.divider()

    # Restart assessment
    if st.button("🔄 Νέα Αξιολόγηση / New Assessment", use_container_width=True):
        st.session_state["answers"] = {q.id: None for q in QUESTIONS}
        st.session_state["step"] = 0
        st.session_state["submitted"] = False
        st.rerun()
