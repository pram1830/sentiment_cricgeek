from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import requests
from pages.auth_pages import (
    show_login_page, show_signup_page, show_verify_email_page, 
    show_forgot_password_page, show_reset_password_page, 
    show_account_settings_page, logout
)
from pages.community_pages import (
    show_community_browser, show_community_detail, show_create_community,
    show_my_communities, show_discover_communities
)
from ui_theme import apply_theme, render_page_header


st.set_page_config(page_title="CricGeek v2.0 - EQS Scoring Dashboard", page_icon="🏏", layout="wide")

# API configuration
API_BASE_URL = "http://localhost:8000/api"

# Initialize session state
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.username = None
    st.session_state.page = "login"
    st.session_state.selected_community = None
    st.session_state.pending_verification_email = None
    st.session_state.ui_light_mode = False


def _render_shell_bar() -> None:
    left_col, right_col = st.columns([7, 1], gap="small")
    mode_label = "Light mode" if st.session_state.get("ui_light_mode", False) else "Dark mode"

    with left_col:
        st.markdown(
            """
            <div class="surface-card" style="display:flex;flex-direction:column;gap:0.2rem;">
                <div class="hero-kicker">CricGeek Platform</div>
                <div style="font-size:1.05rem;font-weight:800;letter-spacing:-0.03em;">EQS scoring, writer communities, and secure auth</div>
                <div class="hero-subtitle" style="margin-top:0.1rem;">Switch between dark and light mode without leaving the app.</div>
                <div class="page-rail">
                    <span class="rail-pill">Blog scoring</span>
                    <span class="rail-pill">Writer communities</span>
                    <span class="rail-pill">Secure auth</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.toggle("Light mode", key="ui_light_mode")
        st.markdown(f'<div class="mini-label" style="text-align:center;margin-top:0.35rem;">{mode_label}</div>', unsafe_allow_html=True)


def _render_feature_strip() -> None:
    mode_label = "Light mode" if st.session_state.get("ui_light_mode", False) else "Dark mode"
    st.markdown(
        f"""
        <div class="feature-strip">
            <div class="feature-card">
                <div class="mini-label">Current mode</div>
                <div class="feature-title">{mode_label}</div>
                <div class="feature-copy">Use the toggle above to switch the whole interface instantly.</div>
            </div>
            <div class="feature-card">
                <div class="mini-label">Blog page</div>
                <div class="feature-title">Readable EQS scoring surface</div>
                <div class="feature-copy">Paste cricket articles into a cleaner, more guided analysis flow.</div>
            </div>
            <div class="feature-card">
                <div class="mini-label">Community page</div>
                <div class="feature-title">Discovery-first writer network</div>
                <div class="feature-copy">Browse communities and same-topic writers in a more polished layout.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_editorial_banner() -> None:
    mode_label = "Light mode" if st.session_state.get("ui_light_mode", False) else "Dark mode"
    st.markdown(
        f"""
        <div class="editorial-banner">
            <div class="editorial-panel">
                <div class="scoreboard-label">Matchday desk</div>
                <div class="scoreboard-value">CricGeek Editorial EQS</div>
                <div class="scoreboard-copy">A cleaner, magazine-style front page for cricket blogs, writer communities, and secure account flows. The same toggle keeps the whole surface readable in both modes.</div>
                <div class="page-rail">
                    <span class="rail-pill">{mode_label}</span>
                    <span class="rail-pill">Blog scoring</span>
                    <span class="rail-pill">Community discovery</span>
                </div>
            </div>
            <div class="editorial-panel">
                <div class="editorial-stats">
                    <div class="editorial-stat">
                        <div class="mini-label">Focus</div>
                        <strong>Readable cricket analysis</strong>
                        <div class="editorial-note">Bigger type, warmer surfaces, and clearer scoring hierarchy.</div>
                    </div>
                    <div class="editorial-stat">
                        <div class="mini-label">Communities</div>
                        <strong>Like-minded writers</strong>
                        <div class="editorial-note">Discovery, matching, and membership in one editorial card stack.</div>
                    </div>
                    <div class="editorial-stat">
                        <div class="mini-label">Mode</div>
                        <strong>Light + dark</strong>
                        <div class="editorial-note">The entire experience shifts without losing contrast or rhythm.</div>
                    </div>
                    <div class="editorial-stat">
                        <div class="mini-label">Style</div>
                        <strong>Sports magazine feel</strong>
                        <div class="editorial-note">A stronger hero, match-day strip, and premium card treatment.</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="accent-rule"></div>
        """,
        unsafe_allow_html=True,
    )


apply_theme("light" if st.session_state.get("ui_light_mode", False) else "dark")
_render_shell_bar()
st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)


SAMPLE_INPUTS: Dict[str, str] = {
    "Select a sample": "",
    "Analytical with stats": (
        "Across the last 12 T20 innings, V Kohli scored 318 runs at a strike rate of 142.8, "
        "compared to 251 at 127.4 before. This suggests an improved powerplay intent and better "
        "boundary conversion under pressure."
    ),
    "Balanced debate": (
        "Some analysts argue an anchor is essential in T20 while others prefer all-out aggression. "
        "However, matchups and venue conditions vary, so a flexible role-based approach is usually "
        "more effective."
    ),
    "Fan emotional": (
        "We love this team and we still believe in them. This loss hurt, but I think they can recover "
        "if they rotate strike better and show intent in the first six overs."
    ),
    "Dismissive complaint": (
        "Same script again. Nothing changes and management never learns. Another frustrating game with "
        "repeated mistakes and no clear plan."
    ),
    "Direct attack": (
        "He is a useless fraud and a disgrace to cricket. Anyone who supports him is clueless."
    ),
}


PROGRESS_STAGES: List[Tuple[int, str]] = [
    (10, "Loading text and preparing input"),
    (25, "Running stance analysis"),
    (40, "Running statistics verification"),
    (60, "Running writing-quality analysis"),
    (80, "Applying calibration"),
    (100, "Final score ready"),
]


@st.cache_resource(show_spinner=False)
def get_pipeline():
    from sentiment_engine.sentiment_pipeline import SentimentPipeline

    return SentimentPipeline()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _display_value(value: Any, precision: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    num = _safe_float(value)
    if num is None:
        return str(value) if str(value).strip() else "N/A"
    return f"{num:.{precision}f}{suffix}"


def _score_color(score: float) -> str:
    if score >= 80.0:
        return "#1f9d55"
    if score >= 60.0:
        return "#2563eb"
    if score >= 40.0:
        return "#f59e0b"
    return "#dc2626"


def _build_explanation(result: Dict[str, Any]) -> Dict[str, List[str] | str]:
    strengths: List[str] = []
    concerns: List[str] = []
    penalties: List[str] = []

    stance_label = str(result.get("stance_label", "N/A"))
    stance_conf = _safe_float(result.get("stance_confidence"))
    stats = result.get("stats_verification", {}) if isinstance(result.get("stats_verification"), dict) else {}
    wq = result.get("writing_quality_breakdown", {}) if isinstance(result.get("writing_quality_breakdown"), dict) else {}
    components = result.get("component_scores", {}) if isinstance(result.get("component_scores"), dict) else {}

    if stance_conf is not None and stance_conf >= 0.5:
        strengths.append(f"Stance detection is confident ({stance_conf:.2f}) with label {stance_label}.")
    else:
        concerns.append("Stance confidence is moderate/low, so interpretation may be less stable.")

    if bool(stats.get("stats_found", False)):
        if bool(stats.get("stats_verified", False)):
            strengths.append("Statistical claims were verified against historical match data.")
        else:
            concerns.append("Statistical claims were found but not fully verified.")
    else:
        concerns.append("No verifiable numeric stats detected in the text.")

    logic = _safe_float(wq.get("argument_logic_score"))
    evidence = _safe_float(wq.get("evidence_presence_score"))
    coherence = _safe_float(wq.get("coherence_score"))
    repetition = _safe_float(wq.get("repetition_penalty"))

    if logic is not None and logic >= 0.5:
        strengths.append("Argument logic connectors are strong.")
    if evidence is not None and evidence >= 0.5:
        strengths.append("Evidence and numeric grounding are present.")
    if coherence is not None and coherence < 0.4:
        concerns.append("Coherence is relatively weak between ideas.")
    if repetition is not None and repetition >= 0.6:
        concerns.append("High repetition lowers writing originality.")

    tox_penalty = _safe_float(components.get("toxicity_penalty"))
    if tox_penalty is not None and tox_penalty < 0:
        penalties.append(f"Toxicity penalty applied ({tox_penalty:.2f}).")

    stat_component = _safe_float(components.get("stat_accuracy_component"))
    if stat_component is not None:
        if stat_component < 0:
            penalties.append("Stats component reduced score due to incorrect claims.")
        elif stat_component > 0:
            strengths.append("Stats component added score due to verified/partially verified claims.")

    archetype = str(result.get("archetype_detected", "N/A"))

    if not strengths:
        strengths.append("Writing shows mixed traits without a dominant positive signal.")
    if not concerns:
        concerns.append("No major concerns were detected by the current deterministic rules.")
    if not penalties:
        penalties.append("No explicit penalties applied beyond baseline shaping.")

    return {
        "archetype": archetype,
        "strengths": strengths,
        "concerns": concerns,
        "penalties": penalties,
    }


def _render_score_band(score: float) -> None:
    color = _score_color(score)
    pct = max(0.0, min(100.0, ((score - 20.0) / 75.0) * 100.0))

    st.markdown(
        f"""
        <div style="margin-top:8px;">
            <div style="font-size:0.9rem;color:var(--cg-muted);margin-bottom:6px;">Score Band (20 to 95)</div>
            <div style="background:var(--cg-chip-bg);border:1px solid var(--cg-border);border-radius:999px;height:14px;overflow:hidden;">
                <div style="width:{pct:.2f}%;background:{color};height:14px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_timeline(stages: List[Tuple[int, str]]) -> None:
    timeline_df = pd.DataFrame(
        [{"Progress": f"{pct}%", "Stage": text, "Status": "Done"} for pct, text in stages]
    )
    st.dataframe(timeline_df, hide_index=True, use_container_width=True)


def _paragraph_table(result: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    paragraphs = result.get("writing_quality_paragraphs", [])
    if not isinstance(paragraphs, list):
        return pd.DataFrame()

    for idx, item in enumerate(paragraphs, start=1):
        signals = item.get("signals", {}) if isinstance(item.get("signals"), dict) else {}
        rows.append(
            {
                "Paragraph": idx,
                "Weight": item.get("weight", "N/A"),
                "Coherence": signals.get("coherence_score", "N/A"),
                "Evidence": signals.get("evidence_presence_score", "N/A"),
                "ArgumentLogic": signals.get("argument_logic_score", "N/A"),
                "RepetitionPenalty": signals.get("repetition_penalty", "N/A"),
                "WQComponent": item.get("writing_quality_component", "N/A"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    render_page_header(
        "EQS Scoring Dashboard",
        "Deterministic cricket-blog scoring with stance, statistics, and writing-quality breakdowns.",
        badges=["EQS", "Qwen 3.5 fallback", "Writer-aware scoring"],
        compact=True,
    )

    _render_editorial_banner()
    _render_feature_strip()

    left_col, right_col = st.columns([1.05, 1.2], gap="large")

    with left_col:
        st.subheader("Input")
        st.markdown(
            "<div class='muted'>Paste a cricket blog/article or choose a sample to see the full analysis path.</div>",
            unsafe_allow_html=True,
        )

        selected_sample = st.selectbox("Quick samples", list(SAMPLE_INPUTS.keys()), index=0)
        default_text = SAMPLE_INPUTS.get(selected_sample, "")

        if "blog_text" not in st.session_state:
            st.session_state.blog_text = default_text
        if selected_sample != "Select a sample" and st.session_state.blog_text != default_text:
            st.session_state.blog_text = default_text

        blog_text = st.text_area(
            "Paste cricket blog/article",
            value=st.session_state.blog_text,
            height=350,
            placeholder="Paste your article here and click Compute EQS...",
        )
        st.session_state.blog_text = blog_text

        st.markdown(
            """
            <div class="soft-panel" style="margin-top:0.9rem;">
                <div class="mini-label">Scoring notes</div>
                <div class="feature-copy">Scoring range: minimum = 20, maximum = 95. The scoring flow is deterministic.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        compute_clicked = st.button("Compute EQS", type="primary", use_container_width=True)

        if compute_clicked:
            if not blog_text.strip():
                st.warning("Please paste a blog/article before computing the score.")
            else:
                progress = st.progress(0)
                status_box = st.empty()
                completed: List[Tuple[int, str]] = []

                for pct, message in PROGRESS_STAGES:
                    status_box.info(f"{pct}% - {message}")
                    progress.progress(pct)
                    completed.append((pct, message))
                    time.sleep(0.28)

                try:
                    pipeline = get_pipeline()
                    result = pipeline.score(blog_text, enable_logs=False)
                    st.session_state.last_result = result
                    st.session_state.last_timeline = completed
                    status_box.success("Computation complete.")
                except Exception as exc:
                    st.session_state.last_result = None
                    st.error(f"Pipeline failed: {exc}")

    with right_col:
        st.subheader("Results")
        result = st.session_state.get("last_result")

        if not result:
            st.markdown("<div class='muted'>Run a scoring pass to view the full dashboard output.</div>", unsafe_allow_html=True)
            return

        score = _safe_float(result.get("final_score")) or 20.0
        score_color = _score_color(score)

        st.markdown(
            f"""
            <div class="score-card editorial-panel">
                <div class="scoreboard-label">Final outcome</div>
                <div style="font-size:0.9rem;color:var(--cg-muted);">Final EQS Score</div>
                <div style="font-size:3rem;font-weight:800;color:{score_color};line-height:1.02;letter-spacing:-0.06em;">{score:.2f}</div>
                <div class="muted">Current range clamp: 20 to 95</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_score_band(score)

        st.markdown("### Compact Status Timeline")
        _render_status_timeline(st.session_state.get("last_timeline", PROGRESS_STAGES))

        st.markdown("### Core Breakdown")
        stats = result.get("stats_verification", {}) if isinstance(result.get("stats_verification"), dict) else {}
        wq = result.get("writing_quality_breakdown", {}) if isinstance(result.get("writing_quality_breakdown"), dict) else {}
        comp = result.get("component_scores", {}) if isinstance(result.get("component_scores"), dict) else {}
        agg = result.get("bqs_aggregation", {}) if isinstance(result.get("bqs_aggregation"), dict) else {}

        card_cols = st.columns(2, gap="small")
        metrics = [
            ("Stance Label", result.get("stance_label", "N/A")),
            ("Stance Confidence", _display_value(result.get("stance_confidence"), 3)),
            ("Stat Accuracy", _display_value(stats.get("stat_accuracy_score"), 3)),
            ("Writing Quality Component", _display_value(comp.get("writing_quality_component"), 2)),
            ("Toxicity Penalty", _display_value(comp.get("toxicity_penalty"), 2)),
            ("Originality Score", _display_value(agg.get("originality_component"), 3)),
            ("Coherence", _display_value(wq.get("coherence_score"), 3)),
            ("Argument Logic", _display_value(wq.get("argument_logic_score"), 3)),
            ("Evidence Presence", _display_value(wq.get("evidence_presence_score"), 3)),
            ("Lexical Diversity", _display_value(wq.get("lexical_diversity_score"), 3)),
            ("Repetition Penalty", _display_value(wq.get("repetition_penalty"), 3)),
        ]

        for idx, (label, value) in enumerate(metrics):
            with card_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="metric-card editorial-stat">
                        <div style="font-size:0.8rem;color:var(--cg-muted);">{label}</div>
                        <div style="font-size:1.1rem;font-weight:600;">{value if str(value).strip() else 'N/A'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Explanation Panel")
        exp = _build_explanation(result)
        st.write(f"Detected writing archetype: **{exp['archetype']}**")
        st.write("Strengths")
        for item in exp["strengths"]:
            st.write(f"- ✅ {item}")
        st.write("Concerns")
        for item in exp["concerns"]:
            st.write(f"- ⚠️ {item}")
        st.write("Penalty Reasons")
        for item in exp["penalties"]:
            st.write(f"- ❌ {item}")

        stats_flag = "Yes" if bool(stats.get("stats_verified", False)) else "No"
        st.caption(f"Stats verified: {stats_flag}")

        st.markdown("### Quality Summary")
        quality_df = pd.DataFrame(
            {
                "Metric": ["Coherence", "Argument Logic", "Evidence", "Lexical Diversity", "Repetition Penalty"],
                "Value": [
                    _safe_float(wq.get("coherence_score")) or 0.0,
                    _safe_float(wq.get("argument_logic_score")) or 0.0,
                    _safe_float(wq.get("evidence_presence_score")) or 0.0,
                    _safe_float(wq.get("lexical_diversity_score")) or 0.0,
                    _safe_float(wq.get("repetition_penalty")) or 0.0,
                ],
            }
        )
        st.bar_chart(quality_df.set_index("Metric"))

        st.markdown("### Soft Fusion Debug")
        soft_debug = agg.get("soft_fusion_debug", {}) if isinstance(agg.get("soft_fusion_debug"), dict) else {}
        soft_metric_cols = st.columns(3, gap="small")
        soft_metrics = [
            ("Top-1 Probability", _display_value(soft_debug.get("top1_probability"), 4)),
            ("Second Probability", _display_value(soft_debug.get("second_probability"), 4)),
            ("Probability Gap", _display_value(soft_debug.get("probability_gap"), 4)),
            ("Dominant Stance Component", _display_value(soft_debug.get("dominant_stance_component"), 2)),
            ("Soft Stance Component", _display_value(soft_debug.get("soft_stance_component"), 2)),
            ("Sentiment Soft Component", _display_value(soft_debug.get("sentiment_soft_component"), 2)),
            ("Final Stance Component", _display_value(soft_debug.get("final_stance_component"), 2)),
            ("Final Core Component", _display_value(soft_debug.get("final_core_component"), 2)),
            ("Dominant Stance", soft_debug.get("dominant_stance", "N/A")),
        ]

        for idx, (label, value) in enumerate(soft_metrics):
            with soft_metric_cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="metric-card editorial-stat">
                        <div style="font-size:0.8rem;color:var(--cg-muted);">{label}</div>
                        <div style="font-size:1.0rem;font-weight:600;">{value if str(value).strip() else 'N/A'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        stance_probs = soft_debug.get("stance_probabilities", {}) if isinstance(soft_debug.get("stance_probabilities"), dict) else {}
        sent_probs = soft_debug.get("sentiment_probabilities", {}) if isinstance(soft_debug.get("sentiment_probabilities"), dict) else {}

        probs_col1, probs_col2 = st.columns(2, gap="small")
        with probs_col1:
            st.caption("Stance Probabilities")
            if stance_probs:
                stance_df = pd.DataFrame(
                    [{"Class": key, "Probability": float(value)} for key, value in stance_probs.items()]
                ).sort_values("Probability", ascending=False)
                st.dataframe(stance_df, hide_index=True, use_container_width=True)
            else:
                st.write("N/A")

        with probs_col2:
            st.caption("Sentiment Probabilities")
            if sent_probs:
                sent_df = pd.DataFrame(
                    [{"Class": key, "Probability": float(value)} for key, value in sent_probs.items()]
                ).sort_values("Probability", ascending=False)
                st.dataframe(sent_df, hide_index=True, use_container_width=True)
            else:
                st.write("N/A")

        warnings = soft_debug.get("warnings", []) if isinstance(soft_debug.get("warnings", []), list) else []
        if warnings:
            st.caption("Fusion Warnings")
            for warning in warnings:
                st.write(f"- {warning}")

        paragraph_df = _paragraph_table(result)
        if not paragraph_df.empty:
            st.markdown("### Paragraph Breakdown")
            st.dataframe(paragraph_df, use_container_width=True, hide_index=True)

        with st.expander("Raw Pipeline Output (JSON)", expanded=False):
            st.code(json.dumps(result, indent=2, default=str), language="json")


# ============================================================================
# PAGE ROUTING
# ============================================================================

def show_main_dashboard():
    """Show the main EQS (Expression Quality Score) scoring dashboard"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.username}!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Dashboard", use_container_width=True):
                st.session_state.page = "dashboard"
                st.rerun()
        with col2:
            if st.button("Logout", use_container_width=True):
                logout()
        
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        if st.button("Score Text (EQS)", use_container_width=True, key="nav_score"):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("My Communities", use_container_width=True, key="nav_communities"):
            st.session_state.page = "my_communities"
            st.rerun()
        
        if st.button("Browse Communities", use_container_width=True, key="nav_browse"):
            st.session_state.page = "community_browser"
            st.rerun()

        if st.button("Discover Writers", use_container_width=True, key="nav_discover"):
            st.session_state.page = "discover_communities"
            st.rerun()

        if st.button("Create Community", use_container_width=True, key="nav_create_community"):
            st.session_state.page = "create_community"
            st.rerun()
        
        if st.button("Settings", use_container_width=True, key="nav_settings"):
            st.session_state.page = "settings"
            st.rerun()
        
        st.divider()
        
        # Rate limit info
        try:
            response = requests.get(
                f"{API_BASE_URL}/rate-limit/status",
                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
                timeout=5
            )
            if response.status_code == 200:
                rate_limit = response.json()
                st.metric(
                    "Qwen 3.5 Usage",
                    f"{rate_limit['request_count']}/{rate_limit['max_requests']}",
                    f"{rate_limit['usage_percentage']:.1f}%"
                )
        except:
            pass
    
    # Main content - EQS Dashboard
    main()


def route_page():
    """Route to appropriate page based on session state"""
    query_page = st.query_params.get("page", "")
    query_token = st.query_params.get("token", "")
    if isinstance(query_page, list):
        query_page = query_page[0] if query_page else ""
    if isinstance(query_token, list):
        query_token = query_token[0] if query_token else ""

    if query_page == "verify_email" or query_token:
        st.session_state.page = "verify_email"
    elif query_page == "reset_password":
        st.session_state.page = "reset_password"

    if not st.session_state.user_logged_in:
        # Authentication pages
        if st.session_state.page == "signup":
            show_signup_page()
        elif st.session_state.page == "verify_email":
            show_verify_email_page()
        elif st.session_state.page == "forgot_password":
            show_forgot_password_page()
        elif st.session_state.page == "reset_password":
            show_reset_password_page()
        else:
            # Default to login
            show_login_page()
    
    else:
        # Logged-in pages
        if st.session_state.page == "dashboard":
            show_main_dashboard()
        elif st.session_state.page == "my_communities":
            show_my_communities(st.session_state.access_token)
        elif st.session_state.page == "community_browser":
            show_community_browser(st.session_state.access_token)
        elif st.session_state.page == "community_detail":
            show_community_detail(st.session_state.selected_community, st.session_state.access_token)
        elif st.session_state.page == "create_community":
            show_create_community(st.session_state.access_token)
        elif st.session_state.page == "discover_communities":
            show_discover_communities(st.session_state.access_token)
        elif st.session_state.page == "settings":
            show_account_settings_page(st.session_state.access_token)
        else:
            # Default to dashboard
            show_main_dashboard()


route_page()
