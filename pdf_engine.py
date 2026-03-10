"""
ContextWeave — pdf_engine.py
Generates a beautiful weekly behavioral report PDF using reportlab.
"""
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import KeepTogether


# ── Colour palette (matches ContextWeave dark theme) ─────────────────────────
C_BG       = colors.HexColor("#050a12")
C_ACCENT   = colors.HexColor("#3b9eff")
C_ACCENT2  = colors.HexColor("#00e5c0")
C_GOLD     = colors.HexColor("#f5c842")
C_TEXT     = colors.HexColor("#dde8f5")
C_MUTED    = colors.HexColor("#5a7090")
C_SURFACE  = colors.HexColor("#0d1520")
C_BORDER   = colors.HexColor("#1a2840")
C_WHITE    = colors.white
C_DANGER   = colors.HexColor("#ff4466")
C_GOOD     = colors.HexColor("#22d98a")


# ── Styles ────────────────────────────────────────────────────────────────────
def _styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=26,
            textColor=C_WHITE, spaceAfter=4, leading=32,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=11,
            textColor=C_MUTED, spaceAfter=0, leading=16,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=10,
            textColor=C_ACCENT, spaceBefore=18, spaceAfter=8,
            letterSpacing=2, leading=14,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=C_TEXT, spaceAfter=6, leading=16,
        ),
        "body_muted": ParagraphStyle(
            "body_muted", fontName="Helvetica", fontSize=9,
            textColor=C_MUTED, spaceAfter=4, leading=14,
        ),
        "metric_val": ParagraphStyle(
            "metric_val", fontName="Helvetica-Bold", fontSize=28,
            textColor=C_WHITE, leading=34, spaceAfter=0,
        ),
        "metric_label": ParagraphStyle(
            "metric_label", fontName="Helvetica", fontSize=8,
            textColor=C_MUTED, leading=12, letterSpacing=1,
        ),
        "insight": ParagraphStyle(
            "insight", fontName="Helvetica", fontSize=9,
            textColor=C_TEXT, spaceAfter=4, leading=15,
            leftIndent=10,
        ),
        "tag": ParagraphStyle(
            "tag", fontName="Helvetica-Bold", fontSize=8,
            textColor=C_ACCENT2, leading=12,
        ),
        "footer": ParagraphStyle(
            "footer", fontName="Helvetica", fontSize=8,
            textColor=C_MUTED, alignment=TA_CENTER, leading=12,
        ),
        "archetype": ParagraphStyle(
            "archetype", fontName="Helvetica-Bold", fontSize=16,
            textColor=C_ACCENT2, spaceAfter=6, leading=22,
        ),
    }


# ── Score colour ──────────────────────────────────────────────────────────────
def _score_color(score):
    if score >= 65: return C_GOOD
    if score >= 40: return C_ACCENT
    return C_DANGER


# ── Metric card (3-up row) ────────────────────────────────────────────────────
def _metric_table(metrics: list, s: dict):
    """
    metrics = [{"label": str, "value": str, "color": Color}, ...]
    """
    cells = []
    for m in metrics:
        val_style = ParagraphStyle(
            "mv", fontName="Helvetica-Bold", fontSize=24,
            textColor=m.get("color", C_WHITE), leading=28,
        )
        lbl_style = ParagraphStyle(
            "ml", fontName="Helvetica", fontSize=8,
            textColor=C_MUTED, leading=11, letterSpacing=1,
        )
        cells.append([
            Paragraph(str(m["value"]), val_style),
            Paragraph(m["label"].upper(), lbl_style),
        ])

    # Build as a row table
    col_data = [[
        Table([[p] for p in cell], colWidths=[55*mm],
              style=TableStyle([
                  ("BACKGROUND", (0,0), (-1,-1), C_SURFACE),
                  ("ROUNDEDCORNERS", [6]),
                  ("BOX", (0,0), (-1,-1), 0.5, C_BORDER),
                  ("TOPPADDING", (0,0), (-1,-1), 10),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                  ("LEFTPADDING", (0,0), (-1,-1), 14),
                  ("RIGHTPADDING", (0,0), (-1,-1), 14),
              ]))
        for cell in cells
    ]]

    t = Table(col_data, colWidths=[60*mm] * len(metrics))
    t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0), (-1,-1), 3),
        ("RIGHTPADDING",  (0,0), (-1,-1), 3),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    return t


# ── Section header ────────────────────────────────────────────────────────────
def _section_header(title: str, s: dict):
    return [
        Spacer(1, 4*mm),
        Paragraph(title, s["section"]),
        HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6),
    ]


# ── Numbered insight list ─────────────────────────────────────────────────────
def _insight_list(items: list, s: dict):
    out = []
    for i, item in enumerate(items, 1):
        clean = item.strip().lstrip("0123456789. ")
        out.append(Paragraph(f"<b>{i}.</b>  {clean}", s["insight"]))
    return out


# ── Main generator ────────────────────────────────────────────────────────────
def generate_report_pdf(
    email: str,
    report_text: str,
    dna: dict,
    notes_count: int,
) -> bytes:
    """
    Generate a styled PDF report.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    s   = _styles()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm,
        title="ContextWeave Behavioral Report",
        author="ContextWeave",
    )

    W = A4[0] - 36*mm   # usable width
    story = []

    # ── Cover header ──────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("🧠 ContextWeave", ParagraphStyle(
            "brand", fontName="Helvetica-Bold", fontSize=13,
            textColor=C_ACCENT, leading=16,
        )),
        Paragraph(
            datetime.now().strftime("%B %d, %Y"),
            ParagraphStyle("date", fontName="Helvetica", fontSize=9,
                           textColor=C_MUTED, alignment=TA_RIGHT, leading=16),
        ),
    ]]
    ht = Table(header_data, colWidths=[W*0.6, W*0.4])
    ht.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,-1), C_SURFACE),
        ("BOX",            (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",     (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 10),
        ("LEFTPADDING",    (0,0), (-1,-1), 14),
        ("RIGHTPADDING",   (0,0), (-1,-1), 14),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(ht)
    story.append(Spacer(1, 6*mm))

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph("Behavioral Intelligence Report", s["title"]))
    handle = email.split("@")[0] if email else "User"
    story.append(Paragraph(f"{handle}  •  {notes_count} notes logged", s["subtitle"]))
    story.append(Spacer(1, 5*mm))

    # ── Core metrics row ──────────────────────────────────────────────────────
    score    = dna.get("avg_score", 50)
    trend    = dna.get("trend", "stable").capitalize()
    streak   = dna.get("streak", 0)
    pattern  = dna.get("dominant_pattern", "—")

    metrics = [
        {"label": "Cognitive Score",  "value": score,   "color": _score_color(score)},
        {"label": "Trend",            "value": trend,   "color": C_ACCENT},
        {"label": f"{streak}-Day Streak", "value": "🔥" if streak >= 3 else str(streak), "color": C_GOLD},
    ]
    story.append(_metric_table(metrics, s))
    story.append(Spacer(1, 4*mm))

    # ── Behavioral Archetype ──────────────────────────────────────────────────
    story += _section_header("BEHAVIORAL ARCHETYPE", s)

    arch_box_data = [[
        Paragraph(dna.get("archetype_emoji","🧠") + "  " + dna.get("archetype","—"), s["archetype"]),
    ],[
        Paragraph(dna.get("archetype_desc",""), s["body_muted"]),
    ],[
        Paragraph(f"Dominant Pattern: {pattern}  •  Top Trait: {dna.get('top_trait','—')}", s["tag"]),
    ]]
    arch_t = Table(arch_box_data, colWidths=[W])
    arch_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_SURFACE),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor("#1a3a5c")),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
    ]))
    story.append(arch_t)

    # ── Parse report text sections ────────────────────────────────────────────
    def _parse_section(header_marker, text):
        """Extract lines after a header marker until the next ── line."""
        lines, capture = [], False
        for line in text.split("\n"):
            if header_marker in line:
                capture = True; continue
            if capture:
                if "━" in line or ("─"*10 in line and line.strip() != line.strip("─")):
                    break
                if line.strip():
                    lines.append(line.strip())
        return lines

    # ── AI Guidance block ─────────────────────────────────────────────────────
    story += _section_header("AI GUIDANCE", s)

    guidance_keys = [
        ("Focus State",    "🎯 Focus State"),
        ("Focus Advice",   "💡 Focus Advice"),
        ("AI Guidance",    "💬 AI Guidance"),
        ("Strategy Mode",  "🧭 Strategy Mode"),
        ("Dominant Risk",  "⚠  Dominant Risk"),
    ]
    for label, marker in guidance_keys:
        for line in report_text.split("\n"):
            if marker in line and ":" in line:
                val = line.split(":", 1)[-1].strip()
                if val:
                    story.append(Paragraph(
                        f"<b>{label}:</b>  {val}", s["body"]
                    ))
                break

    # ── Behavioral Patterns ───────────────────────────────────────────────────
    pat_lines = _parse_section("Behavioral Patterns Detected", report_text)
    if pat_lines:
        story += _section_header("BEHAVIORAL PATTERNS DETECTED", s)
        story += _insight_list(pat_lines, s)

    # ── AI Insights ───────────────────────────────────────────────────────────
    ins_lines = _parse_section("AI Insights", report_text)
    if ins_lines:
        story += _section_header("AI INSIGHTS", s)
        story += _insight_list(ins_lines, s)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec_lines = _parse_section("Recommendations", report_text)
    if rec_lines:
        story += _section_header("RECOMMENDATIONS", s)
        for line in rec_lines:
            story.append(Paragraph("→  " + line.lstrip("•-→ "), s["body"]))

    # ── Weekly insight ────────────────────────────────────────────────────────
    weekly_lines = []
    capture = False
    for line in report_text.split("\n"):
        if "Weekly Behavioral Insight" in line:
            capture = True; continue
        if capture:
            if "━" in line or "📋" in line: break
            if line.strip(): weekly_lines.append(line.strip())

    if weekly_lines:
        story += _section_header("WEEKLY SUMMARY", s)
        for line in weekly_lines:
            story.append(Paragraph(line, s["body_muted"]))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"Generated by ContextWeave Behavioral Intelligence Platform  •  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        s["footer"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()