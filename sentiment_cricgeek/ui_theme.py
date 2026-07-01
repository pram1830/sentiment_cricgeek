from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st


def _theme_tokens(mode: str) -> dict[str, str]:
    if mode == "light":
        return {
            "bg": "#f8f4eb",
            "bg_soft": "#f1eadb",
            "surface": "rgba(255, 255, 255, 0.92)",
            "surface_strong": "rgba(255, 255, 255, 0.98)",
            "border": "rgba(15, 23, 42, 0.10)",
            "text": "#0f172a",
            "muted": "#475569",
            "accent": "#0f766e",
            "accent_2": "#b45309",
            "accent_soft": "rgba(15, 118, 110, 0.10)",
            "chip_bg": "rgba(15, 23, 42, 0.04)",
            "chip_border": "rgba(15, 23, 42, 0.08)",
            "chip_text": "#0f172a",
            "input_bg": "rgba(255, 255, 255, 0.96)",
            "shadow": "0 18px 45px rgba(15, 23, 42, 0.08)",
            "sidebar_start": "rgba(255, 255, 255, 0.98)",
            "sidebar_end": "rgba(244, 247, 252, 0.96)",
        }

    return {
        "bg": "#08111f",
        "bg_soft": "#10243d",
        "surface": "rgba(8, 15, 28, 0.84)",
        "surface_strong": "rgba(11, 21, 38, 0.96)",
        "border": "rgba(148, 163, 184, 0.16)",
        "text": "#eef4ff",
        "muted": "#9eb2cf",
        "accent": "#f59e0b",
        "accent_2": "#22c55e",
        "accent_soft": "rgba(245, 158, 11, 0.16)",
        "chip_bg": "rgba(255, 255, 255, 0.07)",
        "chip_border": "rgba(255, 255, 255, 0.12)",
        "chip_text": "#eff6ff",
        "input_bg": "rgba(6, 12, 22, 0.92)",
        "shadow": "0 24px 60px rgba(2, 6, 23, 0.38)",
        "sidebar_start": "rgba(6, 12, 22, 0.94)",
        "sidebar_end": "rgba(11, 19, 34, 0.96)",
    }


def apply_theme(mode: str) -> None:
    tokens = _theme_tokens(mode)
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

            :root {{
                --cg-bg: {tokens['bg']};
                --cg-bg-soft: {tokens['bg_soft']};
                --cg-surface: {tokens['surface']};
                --cg-surface-strong: {tokens['surface_strong']};
                --cg-border: {tokens['border']};
                --cg-text: {tokens['text']};
                --cg-muted: {tokens['muted']};
                --cg-accent: {tokens['accent']};
                --cg-accent-2: {tokens['accent_2']};
                --cg-accent-soft: {tokens['accent_soft']};
                --cg-chip-bg: {tokens['chip_bg']};
                --cg-chip-border: {tokens['chip_border']};
                --cg-chip-text: {tokens['chip_text']};
                --cg-input-bg: {tokens['input_bg']};
                --cg-shadow: {tokens['shadow']};
                --cg-sidebar-start: {tokens['sidebar_start']};
                --cg-sidebar-end: {tokens['sidebar_end']};
            }}

            html, body, [class*="css"], .stApp {{
                font-family: 'Plus Jakarta Sans', sans-serif;
            }}

            .stApp {{
                color: var(--cg-text);
                background:
                    radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 32%),
                    radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 28%),
                    linear-gradient(180deg, var(--cg-bg) 0%, var(--cg-bg-soft) 100%);
            }}

            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1240px;
            }}

            div[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, var(--cg-sidebar-start), var(--cg-sidebar-end));
                border-right: 1px solid var(--cg-border);
            }}

            div[data-testid="stSidebar"] * {{
                color: var(--cg-text);
            }}

            .hero-card,
            .surface-card,
            .metric-card,
            .score-card,
            .auth-panel,
            .content-card {{
                background: var(--cg-surface);
                border: 1px solid var(--cg-border);
                border-radius: 24px;
                box-shadow: var(--cg-shadow);
                backdrop-filter: blur(20px);
            }}

            .hero-card {{
                padding: 1.2rem 1.4rem;
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }}

            .hero-card.compact {{
                padding: 0.95rem 1.1rem;
            }}

            .hero-card::before {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, var(--cg-accent-soft), transparent 45%);
                pointer-events: none;
            }}

            .hero-card::after {{
                content: '';
                position: absolute;
                left: 0;
                right: 0;
                top: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--cg-accent), var(--cg-accent-2), transparent);
                pointer-events: none;
            }}

            .hero-card > * {{
                position: relative;
                z-index: 1;
            }}

            .hero-kicker {{
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.16em;
                color: var(--cg-muted);
                font-weight: 700;
            }}

            .hero-title {{
                font-size: clamp(1.9rem, 3vw, 3rem);
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: -0.05em;
                margin-top: 0.2rem;
            }}

            .hero-subtitle {{
                color: var(--cg-muted);
                font-size: 0.98rem;
                max-width: 72ch;
                margin-top: 0.45rem;
            }}

            .badge-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.9rem;
            }}

            .badge-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.36rem 0.75rem;
                border-radius: 999px;
                background: var(--cg-chip-bg);
                border: 1px solid var(--cg-chip-border);
                color: var(--cg-chip-text);
                font-size: 0.8rem;
                font-weight: 600;
            }}

            .editorial-banner {{
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 0.9rem;
                margin: 0.9rem 0 1.1rem;
            }}

            .editorial-panel {{
                position: relative;
                padding: 1.1rem 1.15rem;
                border-radius: 22px;
                background: linear-gradient(135deg, var(--cg-surface-strong), rgba(255, 255, 255, 0.03));
                border: 1px solid var(--cg-border);
                box-shadow: var(--cg-shadow);
                overflow: hidden;
            }}

            .editorial-panel::before {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), transparent 55%);
                pointer-events: none;
            }}

            .editorial-panel > * {{
                position: relative;
                z-index: 1;
            }}

            .scoreboard-label {{
                text-transform: uppercase;
                letter-spacing: 0.16em;
                color: var(--cg-muted);
                font-size: 0.72rem;
                font-weight: 800;
            }}

            .scoreboard-value {{
                margin-top: 0.2rem;
                font-size: clamp(1.6rem, 2.3vw, 2.4rem);
                line-height: 1.03;
                font-weight: 800;
                letter-spacing: -0.05em;
            }}

            .scoreboard-copy {{
                margin-top: 0.45rem;
                color: var(--cg-muted);
                max-width: 54ch;
            }}

            .editorial-stats {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.8rem;
            }}

            .editorial-stat {{
                border-radius: 18px;
                border: 1px solid var(--cg-border);
                background: var(--cg-chip-bg);
                padding: 0.85rem 0.95rem;
            }}

            .editorial-stat strong {{
                display: block;
                margin-top: 0.15rem;
                font-size: 1rem;
            }}

            .feature-strip {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.8rem;
                margin: 0.9rem 0 1.1rem;
            }}

            .feature-card {{
                padding: 1rem 1.05rem;
                border-radius: 22px;
                background: linear-gradient(135deg, var(--cg-surface-strong), rgba(245, 158, 11, 0.04));
                border: 1px solid var(--cg-border);
                box-shadow: var(--cg-shadow);
            }}

            .feature-title {{
                font-size: 1.1rem;
                line-height: 1.15;
                font-weight: 800;
                letter-spacing: -0.04em;
                margin-top: 0.2rem;
            }}

            .feature-copy {{
                color: var(--cg-muted);
                font-size: 0.92rem;
                margin-top: 0.3rem;
            }}

            .mini-label {{
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-size: 0.72rem;
                color: var(--cg-muted);
                font-weight: 700;
            }}

            .soft-panel {{
                background: linear-gradient(180deg, var(--cg-surface-strong), var(--cg-surface));
                border: 1px solid var(--cg-border);
                border-radius: 22px;
                box-shadow: var(--cg-shadow);
                padding: 1rem 1.05rem;
            }}

            .soft-panel::before {{
                content: '';
                display: block;
                height: 3px;
                margin: -1rem -1.05rem 0.9rem;
                background: linear-gradient(90deg, var(--cg-accent), var(--cg-accent-2));
                border-radius: 22px 22px 0 0;
            }}

            .detail-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.85rem;
            }}

            .detail-item {{
                border-radius: 18px;
                border: 1px solid var(--cg-border);
                background: var(--cg-chip-bg);
                padding: 0.85rem 0.95rem;
            }}

            .detail-value {{
                display: block;
                margin-top: 0.2rem;
                font-size: 1rem;
                font-weight: 700;
                color: var(--cg-text);
            }}

            .detail-label {{
                font-size: 0.76rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--cg-muted);
                font-weight: 700;
            }}

            .card-chips {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }}

            .card-chip {{
                display: inline-flex;
                align-items: center;
                padding: 0.3rem 0.6rem;
                border-radius: 999px;
                background: var(--cg-chip-bg);
                border: 1px solid var(--cg-chip-border);
                color: var(--cg-chip-text);
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.01em;
            }}

            .card-copy {{
                margin-top: 0.7rem;
                color: var(--cg-muted);
                line-height: 1.55;
            }}

            .page-rail {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.9rem;
            }}

            .rail-pill {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                border-radius: 999px;
                padding: 0.38rem 0.72rem;
                background: var(--cg-chip-bg);
                border: 1px solid var(--cg-chip-border);
                color: var(--cg-chip-text);
                font-size: 0.8rem;
                font-weight: 600;
            }}

            .editorial-row {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1rem;
            }}

            .editorial-note {{
                font-size: 0.92rem;
                color: var(--cg-muted);
                line-height: 1.6;
            }}

            .accent-rule {{
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--cg-border), transparent);
                margin: 1rem 0;
            }}

            .stTabs [data-baseweb="tab-panel"] {{
                padding-top: 0.9rem;
            }}

            div[data-testid="stExpander"] {{
                border-radius: 18px;
                border: 1px solid var(--cg-border);
                background: var(--cg-surface);
                box-shadow: var(--cg-shadow);
            }}

            .section-heading {{
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                margin: 0.35rem 0 0.75rem;
            }}

            .muted {{
                color: var(--cg-muted);
            }}

            .metric-card,
            .score-card {{
                padding: 1rem 1.05rem;
            }}

            .surface-card,
            .auth-panel,
            .content-card {{
                padding: 1rem 1.1rem;
            }}

            .score-card {{
                background: linear-gradient(135deg, var(--cg-surface-strong), var(--cg-surface));
            }}

            div[data-testid="stButton"] > button {{
                background: linear-gradient(135deg, var(--cg-accent), var(--cg-accent-2));
                color: white;
                border: none;
                border-radius: 14px;
                padding: 0.65rem 1rem;
                font-weight: 700;
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.16);
                transition: transform 160ms ease, filter 160ms ease;
            }}

            div[data-testid="stButton"] > button:hover {{
                transform: translateY(-1px);
                filter: brightness(1.03);
            }}

            div[data-testid="stButton"] > button:focus {{
                outline: 2px solid var(--cg-accent);
                outline-offset: 2px;
            }}

            input,
            textarea,
            [data-baseweb="select"] > div,
            [data-baseweb="textarea"] > div {{
                background: var(--cg-input-bg) !important;
                color: var(--cg-text) !important;
                border-color: var(--cg-border) !important;
            }}

            textarea {{
                min-height: 220px !important;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
                border-bottom: 1px solid var(--cg-border);
            }}

            .stTabs [role="tab"] {{
                border-radius: 999px;
                color: var(--cg-muted);
                padding: 0.5rem 0.9rem;
            }}

            .stTabs [role="tab"][aria-selected="true"] {{
                background: var(--cg-surface-strong);
                color: var(--cg-text);
                border: 1px solid var(--cg-border);
            }}

            .stMetric {{
                background: var(--cg-surface);
                border: 1px solid var(--cg-border);
                border-radius: 18px;
                padding: 0.85rem 1rem;
                box-shadow: var(--cg-shadow);
            }}

            .stAlert {{
                border-radius: 18px;
            }}

            .stDataFrame, .stTable {{
                border-radius: 18px;
                overflow: hidden;
            }}

            [data-testid="stSidebar"] .stButton > button {{
                width: 100%;
            }}

            .stMarkdown a {{
                color: var(--cg-accent);
            }}

            .page-divider {{
                height: 1px;
                background: linear-gradient(90deg, transparent, var(--cg-border), transparent);
                margin: 1rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    badges: Sequence[str] | None = None,
    kicker: str = "CricGeek",
    compact: bool = False,
) -> None:
    badge_markup = "".join(f'<span class="badge-pill">{badge}</span>' for badge in badges or [])
    st.markdown(
        f"""
        <div class="hero-card{' compact' if compact else ''}">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="badge-row">{badge_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )