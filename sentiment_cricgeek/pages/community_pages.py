"""
Streamlit community pages for CricGeek writers.
"""

from __future__ import annotations

from html import escape as html_escape
import re
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from ui_theme import render_page_header


API_BASE_URL = "http://localhost:8000/api"

TOPICS = [
    "batting",
    "bowling",
    "fielding",
    "captaincy",
    "strategy",
    "team_performance",
    "player_comparison",
    "tournament",
    "general",
]

WRITING_STYLES = [
    "analytical",
    "balanced",
    "emotional",
    "dismissive",
    "attack_based",
]


def _auth_headers(access_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _error_detail(response: requests.Response, fallback: str) -> str:
    try:
        return response.json().get("detail", fallback)
    except ValueError:
        return fallback


def _suggest_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower().strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:100]


def _render_feature_strip(items: List[Dict[str, str]]) -> None:
    cards = []
    for item in items:
        cards.append(
            f"""
            <div class="feature-card">
                <div class="mini-label">{html_escape(item['label'])}</div>
                <div class="feature-title">{html_escape(item['title'])}</div>
                <div class="feature-copy">{html_escape(item['copy'])}</div>
            </div>
            """
        )

    st.markdown(f'<div class="feature-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_editorial_banner(title: str, copy: str, stats: List[Dict[str, str]]) -> None:
    stat_markup = "".join(
        f"""
        <div class="editorial-stat">
            <div class="mini-label">{html_escape(stat['label'])}</div>
            <strong>{html_escape(stat['value'])}</strong>
            <div class="editorial-note">{html_escape(stat['copy'])}</div>
        </div>
        """
        for stat in stats
    )
    st.markdown(
        f"""
        <div class="editorial-banner">
            <div class="editorial-panel">
                <div class="scoreboard-label">CricGeek community desk</div>
                <div class="scoreboard-value">{html_escape(title)}</div>
                <div class="scoreboard-copy">{html_escape(copy)}</div>
            </div>
            <div class="editorial-panel">
                <div class="editorial-stats">{stat_markup}</div>
            </div>
        </div>
        <div class="accent-rule"></div>
        """,
        unsafe_allow_html=True,
    )


def _community_card(community: Dict[str, Any], *, action_label: str = "View") -> None:
    topic = html_escape(str(community.get("primary_topic", "topic")))
    visibility = html_escape(str(community.get("visibility", "public")).replace("_", " "))
    member_count = community.get("member_count", 0)
    description = html_escape(str(community.get("description") or "No description added yet.")).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="soft-panel editorial-panel">
            <div class="mini-label">Community</div>
            <div class="feature-title">{html_escape(str(community['name']))}</div>
            <div class="card-chips">
                <span class="card-chip">{topic}</span>
                <span class="card-chip">{visibility}</span>
                <span class="card-chip">{member_count} members</span>
            </div>
            <div class="card-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(action_label, key=f"view_{community['id']}", use_container_width=True):
        st.session_state.selected_community = community["id"]
        st.session_state.page = "community_detail"
        st.rerun()


def _render_writer_matches(writers: List[Dict[str, Any]]) -> None:
    if not writers:
        st.info("No same-topic writers found yet.")
        return

    for writer in writers:
        topics = writer.get("primary_topics") or []
        topic_markup = "".join(
            f'<span class="card-chip">{html_escape(str(topic))}</span>' for topic in topics[:4]
        ) or '<span class="card-chip">No topics yet</span>'
        match_reason = html_escape(str(writer.get("match_reason", "similar CricGeek interests"))).replace("\n", "<br>")
        style = html_escape(str(writer.get("writing_style", "unknown")))
        score = writer.get("avg_eqs_score")
        score_text = f"Avg EQS {score:.1f}" if score is not None else f"Reputation {writer.get('reputation_points', 0)}"

        st.markdown(
            f"""
            <div class="soft-panel editorial-panel">
                <div class="mini-label">Writer match</div>
                <div class="feature-title">{html_escape(str(writer['username']))}</div>
                <div class="card-chips">{topic_markup}</div>
                <div class="card-copy">{match_reason}</div>
                <div class="page-rail">
                    <span class="rail-pill">Style: {style}</span>
                    <span class="rail-pill">{html_escape(score_text)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _fetch_writer_matches(
    access_token: str,
    *,
    topic: Optional[str] = None,
    writing_style: Optional[str] = None,
    community_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        params: Dict[str, Any] = {"limit": 20}
        if writing_style and writing_style != "any":
            params["writing_style"] = writing_style

        if community_id:
            url = f"{API_BASE_URL}/communities/{community_id}/same-topic-writers"
        else:
            url = f"{API_BASE_URL}/writers/same-topic"
            if topic and topic != "all":
                params["topic"] = topic

        response = requests.get(
            url,
            params=params,
            headers=_auth_headers(access_token),
            timeout=10,
        )

        if response.status_code == 200:
            return response.json().get("writers", [])

        st.warning(_error_detail(response, "Could not load writer matches"))
        return []
    except requests.exceptions.RequestException as exc:
        st.warning(f"Could not load writer matches: {exc}")
        return []


def show_community_browser(access_token: str):
    """Browse public communities and same-topic writers."""
    render_page_header(
        "Community Browser",
        "Discover public communities and like-minded writers by cricket topic or writing style.",
        badges=["Discover", "Same-topic writers", "Public communities"],
        compact=True,
    )

    _render_editorial_banner(
        "Community Browser",
        "A magazine-style directory for CricGeek communities, filtered by topic and writer style.",
        [
            {"label": "Spaces", "value": "Public communities", "copy": "Open groups ready for discovery and joining."},
            {"label": "Writers", "value": "Same-topic matches", "copy": "Find writers with overlapping cricket interests."},
            {"label": "View", "value": "Editorial cards", "copy": "Each result is surfaced as a clean matchday panel."},
            {"label": "Filter", "value": "Topic + style", "copy": "Search by theme without losing the larger context."},
        ],
    )

    _render_feature_strip([
        {"label": "Browse", "title": "Public spaces", "copy": "Scan open communities, compare topics, and jump into the right space."},
        {"label": "Match", "title": "Same-topic writers", "copy": "Find people writing on the same cricket themes and styles."},
        {"label": "Filter", "title": "Search and sort", "copy": "Narrow by topic, search term, or popularity without losing context."},
    ])

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            topic_filter = st.selectbox("Topic", options=["all"] + TOPICS)
        with col2:
            search = st.text_input("Search communities")
        with col3:
            sort_by = st.selectbox("Sort by", ["Recent", "Popular"])

    tab_communities, tab_writers = st.tabs(["Communities", "Same-topic writers"])

    with tab_communities:
        try:
            params: Dict[str, Any] = {"limit": 50}
            if topic_filter != "all":
                params["topic"] = topic_filter
            if search:
                params["search"] = search

            response = requests.get(
                f"{API_BASE_URL}/communities",
                params=params,
                headers=_auth_headers(access_token),
                timeout=10,
            )

            if response.status_code != 200:
                st.error(_error_detail(response, "Failed to load communities"))
                return

            communities = response.json().get("communities", [])
            st.markdown(
                f"<div class='muted' style='margin:0.25rem 0 0.8rem;'>Showing {len(communities)} community result(s).</div>",
                unsafe_allow_html=True,
            )
            if sort_by == "Popular":
                communities.sort(key=lambda item: item.get("member_count", 0), reverse=True)

            if not communities:
                st.info("No communities found.")
            else:
                for community in communities:
                    _community_card(community)

        except requests.exceptions.RequestException as exc:
            st.error(f"Connection error: {exc}")

    with tab_writers:
        style_filter = st.selectbox("Writing style", options=["any"] + WRITING_STYLES)
        writers = _fetch_writer_matches(
            access_token,
            topic=topic_filter,
            writing_style=style_filter,
        )
        _render_writer_matches(writers)


def show_community_detail(community_id: str, access_token: str):
    """Show community details."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/communities/{community_id}",
            headers=_auth_headers(access_token),
            timeout=10,
        )

        if response.status_code != 200:
            st.error(_error_detail(response, "Community not found"))
            return

        community = response.json()

        render_page_header(
            community["name"],
            community.get("description") or "Community detail page for membership and discovery.",
            badges=[community.get("primary_topic", "topic"), f"{community.get('member_count', 0)} members", community.get("visibility", "public")],
            compact=True,
        )

        _render_editorial_banner(
            community["name"],
            community.get("description") or "A focused community detail view built for discovery, membership, and writer matching.",
            [
                {"label": "Topic", "value": str(community.get("primary_topic", "topic")), "copy": "Primary cricket theme for this community."},
                {"label": "Members", "value": f"{community.get('member_count', 0)} writers", "copy": "Current writers inside the space."},
                {"label": "Visibility", "value": str(community.get("visibility", "public")).replace("_", " "), "copy": "How openly the community can be joined."},
                {"label": "Created", "value": str(community['created_at'][:10]), "copy": "The community launch date."},
            ],
        )

        _render_feature_strip([
            {"label": "Topic", "title": str(community.get("primary_topic", "topic")), "copy": "The primary topic that anchors this community."},
            {"label": "Members", "title": f"{community.get('member_count', 0)} writers", "copy": "People already active in this space."},
            {"label": "Visibility", "title": str(community.get("visibility", "public")).replace("_", " "), "copy": "How open this space is for discovery and joining."},
        ])

        col1, col2 = st.columns([2, 1])
        with col1:
            with st.container(border=True):
                st.caption(f"{community['primary_topic']} | Created {community['created_at'][:10]}")
                if community.get("description"):
                    st.write(community["description"])

        with col2:
            with st.container(border=True):
                st.metric("Members", community["member_count"])
                if st.button("Join Community", use_container_width=True):
                    join_response = requests.post(
                        f"{API_BASE_URL}/communities/{community_id}/join",
                        headers=_auth_headers(access_token),
                        timeout=10,
                    )
                    if join_response.status_code == 200:
                        st.success("Joined community.")
                        st.rerun()
                    else:
                        st.error(_error_detail(join_response, "Failed to join community"))

        tab_members, tab_writers = st.tabs(["Members", "Same-topic writers"])

        with tab_members:
            members_response = requests.get(
                f"{API_BASE_URL}/communities/{community_id}/members",
                params={"limit": 20},
                headers=_auth_headers(access_token),
                timeout=10,
            )

            if members_response.status_code == 200:
                members = members_response.json().get("members", [])
                if not members:
                    st.info("No members yet.")
                for member in members:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(member["username"])
                    with col2:
                        st.caption(member["role"])
                    with col3:
                        st.caption(member["joined_at"][:10])
            else:
                st.warning(_error_detail(members_response, "Could not load members"))

        with tab_writers:
            style_filter = st.selectbox("Writing style", options=["any"] + WRITING_STYLES)
            writers = _fetch_writer_matches(
                access_token,
                community_id=community_id,
                writing_style=style_filter,
            )
            _render_writer_matches(writers)

    except requests.exceptions.RequestException as exc:
        st.error(f"Connection error: {exc}")


def show_create_community(access_token: str):
    """Create a community for like-minded CricGeek writers."""
    render_page_header(
        "Create Writer Community",
        "Build a space for writers who care about the same cricket topic and expression style.",
        badges=["Public", "Private", "Invite-only"],
        compact=True,
    )

    _render_editorial_banner(
        "Create Community",
        "Set the theme, audience, and visibility for a new writer space in one place.",
        [
            {"label": "Name", "value": "Identity first", "copy": "Choose a name and slug that feels publishable."},
            {"label": "Topics", "value": "Primary + related", "copy": "Anchor the community around a cricket conversation."},
            {"label": "Access", "value": "Visibility rules", "copy": "Decide who can join before the room goes live."},
            {"label": "Profile", "value": "Writer style sync", "copy": "Update your own writing profile alongside the community."},
        ],
    )

    _render_feature_strip([
        {"label": "Name it", "title": "Set the identity", "copy": "Choose a memorable community name and a clean slug."},
        {"label": "Frame it", "title": "Topic + style", "copy": "Pin the cricket topic and the writing style the group should attract."},
        {"label": "Control it", "title": "Visibility rules", "copy": "Pick public, private, or invite-only before publishing."},
    ])

    with st.container(border=True):
        with st.form("create_community_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Community Name", max_chars=100)
            with col2:
                manual_slug = st.text_input("Community Slug", help="Lowercase URL slug, optional")

            description = st.text_area(
                "Description",
                help="Explain the topic, tone, and kind of writing this community is for.",
                max_chars=1000,
            )

            col1, col2 = st.columns(2)
            with col1:
                primary_topic = st.selectbox("Primary Topic", options=TOPICS)
                writer_style = st.selectbox("Your Writing Style", options=WRITING_STYLES)
            with col2:
                secondary_topics = st.multiselect("Related Topics", options=TOPICS, max_selections=3)
                visibility = st.radio(
                    "Visibility",
                    options=["public", "private", "invite_only"],
                    horizontal=True,
                )

            submitted = st.form_submit_button("Create Community", use_container_width=True)

            if submitted:
                slug = manual_slug.strip().lower() or _suggest_slug(name)
                if not name:
                    st.error("Name is required")
                elif len(name) < 3:
                    st.error("Name must be at least 3 characters")
                elif not slug:
                    st.error("Slug is required")
                elif not description:
                    st.error("Description is required")
                else:
                    topics = [primary_topic] + [topic for topic in secondary_topics if topic != primary_topic]

                    try:
                        profile_response = requests.put(
                            f"{API_BASE_URL}/users/me/writer-profile",
                            headers=_auth_headers(access_token),
                            json={
                                "writing_style": writer_style,
                                "primary_topics": topics[:5],
                            },
                            timeout=10,
                        )
                        if profile_response.status_code != 200:
                            st.warning(_error_detail(profile_response, "Could not update writer profile"))

                        response = requests.post(
                            f"{API_BASE_URL}/communities",
                            headers=_auth_headers(access_token),
                            json={
                                "name": name,
                                "slug": slug,
                                "description": description,
                                "primary_topic": primary_topic,
                                "secondary_topics": secondary_topics,
                                "visibility": visibility,
                            },
                            timeout=10,
                        )

                        if response.status_code == 200:
                            st.success("Community created.")
                            st.balloons()
                            time.sleep(1)
                            st.session_state.page = "my_communities"
                            st.rerun()
                        else:
                            st.error(_error_detail(response, "Creation failed"))

                    except requests.exceptions.RequestException as exc:
                        st.error(f"Connection error: {exc}")

    if st.button("Cancel"):
        st.session_state.page = "my_communities"
        st.rerun()


def show_my_communities(access_token: str):
    """Show communities current user is a member of."""
    render_page_header(
        "My Communities",
        "Your current memberships and the writer communities you already belong to.",
        badges=["Memberships", "Current space", "Writer network"],
        compact=True,
    )

    _render_editorial_banner(
        "My Communities",
        "A tidy roster of the writer spaces you already follow, with quick access to browse, discover, or create more.",
        [
            {"label": "Current", "value": "Joined spaces", "copy": "The communities you already belong to."},
            {"label": "Next", "value": "Discovery path", "copy": "Use browse or discover to expand your network."},
            {"label": "Action", "value": "Create space", "copy": "Launch a new room when the topic is missing."},
            {"label": "Layout", "value": "Editorial roster", "copy": "A stronger hierarchy for membership views."},
        ],
    )

    _render_feature_strip([
        {"label": "Memberships", "title": "Your active spaces", "copy": "Jump back into communities you already belong to."},
        {"label": "Browse", "title": "Find new spaces", "copy": "Move quickly to discovery or create a fresh community."},
        {"label": "Network", "title": "Stay organized", "copy": "Keep your writer spaces grouped in one clean view."},
    ])

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Browse", use_container_width=True):
            st.session_state.page = "community_browser"
            st.rerun()
    with col2:
        if st.button("Discover", use_container_width=True):
            st.session_state.page = "discover_communities"
            st.rerun()
    with col3:
        if st.button("Create", use_container_width=True):
            st.session_state.page = "create_community"
            st.rerun()

    try:
        response = requests.get(
            f"{API_BASE_URL}/users/me/communities",
            headers=_auth_headers(access_token),
            timeout=10,
        )

        if response.status_code != 200:
            st.error(_error_detail(response, "Failed to load communities"))
            return

        communities = response.json().get("communities", [])
        st.divider()

        if not communities:
            st.info("You have not joined any communities yet.")
            return

        for community in communities:
            _community_card(community)

    except requests.exceptions.RequestException as exc:
        st.error(f"Connection error: {exc}")


def show_discover_communities(access_token: str):
    """Show recommended communities and writers."""
    render_page_header(
        "Discover",
        "Recommended communities and like-minded writers based on your topics and style.",
        badges=["Recommendations", "Topic match", "Style match"],
        compact=True,
    )

    _render_editorial_banner(
        "Discover",
        "Recommendations presented like a feature page, with a cleaner path to people and communities that fit your voice.",
        [
            {"label": "Match", "value": "Recommended spaces", "copy": "Communities aligned with your profile."},
            {"label": "People", "value": "Like-minded writers", "copy": "Find writers who share a similar tone or topic."},
            {"label": "Join", "value": "One-click entry", "copy": "Move from discovery to membership quickly."},
            {"label": "Feel", "value": "Sports desk style", "copy": "A cleaner, more editorial front page treatment."},
        ],
    )

    _render_feature_strip([
        {"label": "Recommended", "title": "Curated spaces", "copy": "See communities that align with your topics and style."},
        {"label": "Writers", "title": "Like-minded people", "copy": "Discover people who write in a similar cricket voice."},
        {"label": "Action", "title": "Join fast", "copy": "Move from discovery to membership without extra friction."},
    ])

    tab_communities, tab_writers = st.tabs(["Recommended communities", "Like-minded writers"])

    with tab_communities:
        try:
            response = requests.get(
                f"{API_BASE_URL}/communities/discover",
                headers=_auth_headers(access_token),
                timeout=10,
            )

            if response.status_code != 200:
                st.error(_error_detail(response, "Failed to load recommendations"))
            else:
                communities = response.json().get("communities", [])
                if not communities:
                    st.info("No recommendations available yet.")
                for community in communities:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(community["name"])
                            st.caption(f"{community['primary_topic']} | {community['member_count']} members")
                            if community.get("description"):
                                st.write(community["description"][:180])
                        with col2:
                            if st.button("Join", key=f"join_{community['id']}", use_container_width=True):
                                join_response = requests.post(
                                    f"{API_BASE_URL}/communities/{community['id']}/join",
                                    headers=_auth_headers(access_token),
                                    timeout=10,
                                )
                                if join_response.status_code == 200:
                                    st.success("Joined.")
                                    st.rerun()
                                else:
                                    st.error(_error_detail(join_response, "Failed to join"))

        except requests.exceptions.RequestException as exc:
            st.error(f"Connection error: {exc}")

    with tab_writers:
        col1, col2 = st.columns(2)
        with col1:
            topic_filter = st.selectbox("Topic", options=["all"] + TOPICS)
        with col2:
            style_filter = st.selectbox("Writing style", options=["any"] + WRITING_STYLES)

        writers = _fetch_writer_matches(
            access_token,
            topic=topic_filter,
            writing_style=style_filter,
        )
        _render_writer_matches(writers)
