import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------
# Data model
# ---------------------------

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
    }
}

# 4 questions per domain (v1)
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


# ---------------------------
# Language strings
# ---------------------------

UI = {
    "GR": {
        "app_title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "intro_title": "Αυτοαξιολόγηση (Self-completed)",
        "intro_body": (
            "Συμπληρώστε την αξιολόγηση με βάση την πραγματική κατάσταση. "
            "Στο τέλος θα δείτε συνοπτικό dashboard, δείκτες ωριμότητας, συγκέντρωση κινδύνων και προτεραιότητες."
        ),
        "scale_title": "Κλίμακα Ωριμότητας 1–5 (Ορισμοί)",
        "scale": {
            1: "Άτυπο / Αποσπασματικό: εξάρτηση από πρόσωπα, χωρίς σταθερή δομή ή τεκμηρίωση.",
            2: "Μερικώς Ορισμένο: υπάρχουν πρακτικές αλλά ασυνέπεια/επιλεκτική εφαρμογή.",
            3: "Ορισμένο αλλά όχι πλήρως ενσωματωμένο: δομές υπάρχουν, η εφαρμογή δεν είναι σταθερή.",
            4: "Ενσωματωμένο & αποτελεσματικό: σαφές, συνεπές, υποστηρίζει ποιοτικές αποφάσεις.",
            5: "Προηγμένο / Πρότυπο: πλήρως ενσωματωμένο, με συστηματική αναθεώρηση και υψηλή ωριμότητα."
        },
        "start": "Ξεκινήστε την αξιολόγηση",
        "domain_tab": "Ενότητα",
        "question_help": "Επιλέξτε βαθμό 1–5.",
        "results": "Αποτελέσματα",
        "overall_index": "Συνολικός Δείκτης Ωριμότητας (0–100)",
        "priority_title": "Κορυφαίες Προτεραιότητες (Top Focus Areas)",
        "download": "Λήψη αποτελεσμάτων (CSV)",
        "incomplete": "Υπάρχουν ερωτήσεις χωρίς απάντηση. Παρακαλώ συμπληρώστε όλες τις ερωτήσεις.",
        "interpretations": "Ερμηνεία & Επιπτώσεις Συζήτησης",
        "overall_interp_title": "Συνοπτική Ερμηνεία",
        "risk_matrix": "Χάρτης Συγκέντρωσης Κινδύνου (Score × Weight)",
        "radar": "Radar Ωριμότητας",
        "bars": "Ανά Ενότητα (Μέσος Όρος 1–5)",
    },
    "EN": {
        "app_title": "Legacy360° | Family Governance & Succession Roadmap",
        "tagline": "a Strategize service",
        "intro_title": "Self-completed assessment",
        "intro_body": (
            "Complete the assessment based on current reality. "
            "At the end you will receive a dashboard with maturity scores, risk concentration and priorities."
        ),
        "scale_title": "Maturity Scale 1–5 (Anchors)",
        "scale": {
            1: "Informal / ad-hoc: person-dependent, no consistent structure or documentation.",
            2: "Partially defined: some practices exist but inconsistent / selectively applied.",
            3: "Defined but not embedded: structures exist; adoption and compliance vary.",
            4: "Embedded & effective: clearly defined and consistently applied; supports decision quality.",
            5: "Advanced / role model: fully embedded, continuously reviewed; maturity beyond peers."
        },
        "start": "Start assessment",
        "domain_tab": "Domain",
        "question_help": "Select a score 1–5.",
        "results": "Results",
        "overall_index": "Overall Maturity Index (0–100)",
        "priority_title": "Top Focus Areas",
        "download": "Download results (CSV)",
        "incomplete": "Some questions are unanswered. Please complete all questions.",
        "interpretations": "Interpretation & Discussion Implications",
        "overall_interp_title": "Executive Summary Interpretation",
        "risk_matrix": "Risk Concentration Map (Score × Weight)",
        "radar": "Maturity Radar",
        "bars": "By Domain (Average 1–5)",
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

# Simple domain interpretation templates by band (v1)
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
        "RED": "Το συνολικό προφίλ ωριμότητας υποδηλώνει υψηλό διακυβερνητικό και εκτελεστικό κίνδυνο. Συνιστάται άμεση εστίαση στα κρίσιμα πεδία πριν από μεγάλες δεσμεύσεις (επενδύσεις, διαδοχή, εξωτερική ανάπτυξη).",
        "AMBER": "Υπάρχει λειτουργική βάση, αλλά η ωριμότητα δεν είναι ακόμη συστηματικά ενσωματωμένη. Με στοχευμένες παρεμβάσεις σε υψηλού βάρους ενότητες, η επιχείρηση μπορεί να μειώσει σημαντικά τον κίνδυνο και να ενισχύσει τη συνέχεια.",
        "GREEN": "Το προφίλ δείχνει ισχυρή ωριμότητα. Προτεραιότητα: διατήρηση πειθαρχίας, περιοδικές αναθεωρήσεις και προληπτική προετοιμασία διαδοχής/συνέχειας.",
    },
    "EN": {
        "RED": "The overall maturity profile indicates elevated governance and execution risk. Prioritise critical areas before major commitments (investments, succession moves, external expansion).",
        "AMBER": "A functional base exists, but maturity is not yet consistently embedded. Targeted interventions in high-weight domains can materially reduce risk and strengthen continuity.",
        "GREEN": "The profile indicates strong maturity. Maintain discipline, run periodic reviews and proactively prepare succession/continuity.",
    }
}


# ---------------------------
# Scoring helpers
# ---------------------------

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
    # Convert 1–5 to 0–100
    # 1 => 0, 5 => 100
    return (total - 1.0) / 4.0 * 100.0

def risk_priority(domain_key: str, score: float, weight: float) -> float:
    # Higher risk when score is low and weight is high
    # Normalise: risk = (6 - score) * weight
    return (6.0 - score) * weight

def make_radar(labels: List[str], values: List[float], title: str):
    # Close the loop
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


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Legacy360°", layout="wide")

# Language selector (Greek primary)
lang = st.sidebar.radio("Language / Γλώσσα", ["GR", "EN"], index=0)

# Header (logo placeholders)
left, right = st.columns([0.75, 0.25], vertical_alignment="center")
with left:
    st.title(UI[lang]["app_title"])
    st.caption(UI[lang]["tagline"])
with right:
    # Placeholder for logos:
    st.markdown("**[Legacy360° Logo Placeholder]**  \n*a Strategize service*")

st.divider()

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

# Build question groups per domain
domain_questions: Dict[str, List[Question]] = {d.key: [] for d in DOMAINS}
for q in QUESTIONS:
    domain_questions[q.domain_key].append(q)

tabs = st.tabs([f"🧭 {DOMAIN_LABELS[lang][d.key]}" for d in DOMAINS] + [f"📊 {UI[lang]['results']}"])

# Session state init
if "answers" not in st.session_state:
    st.session_state["answers"] = {}  # question_id -> score

def render_domain_tab(domain: Domain, tab):
    with tab:
        st.markdown(f"### {DOMAIN_LABELS[lang][domain.key]}")
        st.caption(f"Weight / Βάρος: **{int(domain.weight*100)}%**")
        st.write("")
        for q in domain_questions[domain.key]:
            key = f"ans_{q.id}"
            default = st.session_state["answers"].get(q.id, 3)
            score = st.radio(
                label=f"**{q.id}** — {q.text[lang]}",
                options=[1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(default),
                horizontal=True,
                help=UI[lang]["question_help"],
                key=key
            )
            st.session_state["answers"][q.id] = score
            st.write("")

# Render domain tabs
for i, d in enumerate(DOMAINS):
    render_domain_tab(d, tabs[i])

# Results tab
with tabs[-1]:
    st.markdown(f"## {UI[lang]['results']}")

    # Validate completeness
    all_ids = [q.id for q in QUESTIONS]
    missing = [qid for qid in all_ids if qid not in st.session_state["answers"]]
    if missing:
        st.error(UI[lang]["incomplete"])
        st.stop()

    # Compute domain averages
    domain_scores: Dict[str, float] = {}
    rows = []
    for d in DOMAINS:
        qs = domain_questions[d.key]
        vals = [st.session_state["answers"][q.id] for q in qs]
        avg = float(np.mean(vals))
        domain_scores[d.key] = avg
        rows.append({
            "domain_key": d.key,
            "domain": DOMAIN_LABELS[lang][d.key],
            "weight": d.weight,
            "avg_score": avg,
            "band": band_for_score(avg),
            "risk": risk_priority(d.key, avg, d.weight),
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
        # Highest risk domain label
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
        bar_df = pd.DataFrame({
            "Domain": labels,
            "Avg (1–5)": values,
        })
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

    # Risk map (heatmap-like table)
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
    top_n = 5
    pri = df.head(top_n)

    for _, r in pri.iterrows():
        dom_key = r["domain_key"]
        band = r["band"]
        st.markdown(f"### {'🔴' if band=='RED' else '🟡' if band=='AMBER' else '🟢'} {DOMAIN_LABELS[lang][dom_key]}")
        st.caption(f"Weight / Βάρος: {int(r['weight']*100)}% · Avg: {r['avg_score']:.2f} · {BAND_LABELS[lang][band]}")
        st.write(DOMAIN_INTERP[lang][band])

    st.divider()

    # Interpretations
    st.subheader(UI[lang]["interpretations"])
    overall_band = band_for_score(float(np.mean(list(domain_scores.values()))))
    st.markdown(f"### {UI[lang]['overall_interp_title']}")
    st.write(OVERALL_INTERP[lang][overall_band])

    st.divider()

    # Download CSV
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

    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(UI[lang]["download"], data=csv, file_name="legacy360_results.csv", mime="text/csv")
