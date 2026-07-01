"""
Streamlit auth pages for login, signup, email verification, and password reset.
"""

import time

import requests
import streamlit as st

from ui_theme import render_page_header


API_BASE_URL = "http://localhost:8000/api"


def _error_detail(response: requests.Response, fallback: str) -> str:
    try:
        return response.json().get("detail", fallback)
    except ValueError:
        return fallback


def show_signup_page():
    """Signup page."""
    render_page_header(
        "Create CricGeek Account",
        "Secure signup with email verification, encrypted password storage, and JWT session tokens.",
        badges=["Email verification", "bcrypt", "JWT"],
        compact=True,
    )

    with st.container(border=True):
        with st.form("signup_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                username = st.text_input(
                    "Username",
                    help="3-50 characters, alphanumeric + - _",
                )

            with col2:
                email = st.text_input("Email", type="default")

            password = st.text_input(
                "Password",
                type="password",
                help="At least 8 characters",
            )

            password_confirm = st.text_input(
                "Confirm Password",
                type="password",
            )

            submitted = st.form_submit_button("Create Account", use_container_width=True)

            if submitted:
                if not username:
                    st.error("Username is required")
                elif len(username) < 3 or len(username) > 50:
                    st.error("Username must be 3-50 characters")
                elif not email:
                    st.error("Email is required")
                elif password != password_confirm:
                    st.error("Passwords do not match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/signup",
                            json={
                                "username": username,
                                "email": email,
                                "password": password,
                            },
                            timeout=10,
                        )

                        if response.status_code == 200:
                            st.session_state.pending_verification_email = email
                            st.session_state.access_token = None
                            st.session_state.refresh_token = None
                            st.session_state.user_logged_in = False
                            st.session_state.username = None
                            st.session_state.page = "verify_email"
                            st.success("Account created.")
                            st.info("Please verify your email before logging in.")
                            time.sleep(1)
                            st.rerun()

                        st.error(_error_detail(response, "Signup failed"))

                    except requests.exceptions.RequestException as exc:
                        st.error(f"Connection error: {exc}")

    if st.button("Already have an account? Login"):
        st.session_state.page = "login"
        st.rerun()


def show_login_page():
    """Login page."""
    render_page_header(
        "Login to CricGeek",
        "Pick up where you left off with secure JWT authentication and verified accounts.",
        badges=["JWT login", "Verified users", "Secure session"],
        compact=True,
    )

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        with st.container(border=True):
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)

                if submitted:
                    if not username or not password:
                        st.error("Username and password are required")
                    else:
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/auth/login",
                                json={
                                    "username": username,
                                    "password": password,
                                },
                                timeout=10,
                            )

                            if response.status_code == 200:
                                data = response.json()
                                st.session_state.access_token = data["access_token"]
                                st.session_state.refresh_token = data["refresh_token"]
                                st.session_state.user_logged_in = True
                                st.session_state.username = username
                                st.session_state.page = "dashboard"
                                st.success("Logged in successfully.")
                                time.sleep(0.5)
                                st.rerun()

                            error_detail = _error_detail(response, "Login failed")
                            st.error(error_detail)
                            if "verify" in error_detail.lower():
                                st.session_state.pending_verification_email = username if "@" in username else ""
                                st.info("Open the verification page to paste your token or resend the email.")

                        except requests.exceptions.RequestException as exc:
                            st.error(f"Connection error: {exc}")

    with col2:
        with st.container(border=True):
            st.subheader("New to CricGeek?")
            st.caption("Create an account, verify your email, and join writer communities.")
            if st.button("Create Account", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()
            if st.button("Verify Email", use_container_width=True):
                st.session_state.page = "verify_email"
                st.rerun()
            if st.button("Forgot Password", use_container_width=True):
                st.session_state.page = "forgot_password"
                st.rerun()


def show_verify_email_page():
    """Email verification page."""
    render_page_header(
        "Verify Your Email",
        "Complete registration by confirming the token sent to your inbox.",
        badges=["Token verification", "24 hour expiry", "Secure onboarding"],
        compact=True,
    )

    st.info("Check your email for a verification link or paste the token below.")

    query_token = st.query_params.get("token", "")
    if isinstance(query_token, list):
        query_token = query_token[0] if query_token else ""

    with st.container(border=True):
        with st.form("verify_email_form"):
            token = st.text_input("Verification Token", value=query_token, type="password")
            submitted = st.form_submit_button("Verify Email", use_container_width=True)

            if submitted:
                if not token:
                    st.error("Token is required")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/verify-email",
                            json={"token": token},
                            timeout=10,
                        )

                        if response.status_code == 200:
                            st.success("Email verified. You can now log in.")
                            st.session_state.page = "login"
                            st.query_params.clear()
                            st.balloons()
                            time.sleep(1)
                            st.rerun()

                        st.error(_error_detail(response, "Verification failed"))

                    except requests.exceptions.RequestException as exc:
                        st.error(f"Connection error: {exc}")

    st.divider()
    st.subheader("Need a new verification email?")

    default_email = st.session_state.get("pending_verification_email", "")
    with st.form("resend_verification_form"):
        email = st.text_input("Email Address", value=default_email, type="default")
        resend_submitted = st.form_submit_button("Resend Verification Email", use_container_width=True)

        if resend_submitted:
            if not email:
                st.error("Email is required")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/auth/resend-verification",
                        json={"email": email},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        st.session_state.pending_verification_email = email
                        st.success(response.json().get("message", "Verification email sent."))
                    else:
                        st.error(_error_detail(response, "Request failed"))

                except requests.exceptions.RequestException as exc:
                    st.error(f"Connection error: {exc}")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def show_forgot_password_page():
    """Password reset request page."""
    render_page_header(
        "Reset Password",
        "Request a password reset email for your CricGeek account.",
        badges=["Account recovery", "Email delivery", "Secure reset"],
        compact=True,
    )

    with st.container(border=True):
        with st.form("forgot_password_form"):
            email = st.text_input("Email Address", type="default")
            submitted = st.form_submit_button("Send Reset Link", use_container_width=True)

            if submitted:
                if not email:
                    st.error("Email is required")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/forgot-password",
                            json={"email": email},
                            timeout=10,
                        )

                        if response.status_code == 200:
                            st.success("If the email exists, reset instructions have been sent.")
                            st.info("Check your email for reset instructions.")
                        else:
                            st.error(_error_detail(response, "Request failed"))

                    except requests.exceptions.RequestException as exc:
                        st.error(f"Connection error: {exc}")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def show_reset_password_page():
    """Password reset confirmation page."""
    render_page_header(
        "Reset Your Password",
        "Use the token from your email to create a new secure password.",
        badges=["Token based", "8+ characters", "Secure recovery"],
        compact=True,
    )

    query_token = st.query_params.get("token", "")
    if isinstance(query_token, list):
        query_token = query_token[0] if query_token else ""

    with st.container(border=True):
        with st.form("reset_password_form"):
            token = st.text_input("Reset Token from Email", value=query_token, type="password")
            new_password = st.text_input("New Password", type="password", help="At least 8 characters")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Reset Password", use_container_width=True)

            if submitted:
                if not token:
                    st.error("Reset token is required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/auth/reset-password",
                            json={
                                "token": token,
                                "new_password": new_password,
                            },
                            timeout=10,
                        )

                        if response.status_code == 200:
                            st.success("Password reset successfully. You can now log in.")
                            st.session_state.page = "login"
                            st.query_params.clear()
                            time.sleep(1)
                            st.rerun()

                        st.error(_error_detail(response, "Reset failed"))

                    except requests.exceptions.RequestException as exc:
                        st.error(f"Connection error: {exc}")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def logout():
    """Logout user."""
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_logged_in = False
    st.session_state.username = None
    st.session_state.page = "login"
    st.rerun()


def show_account_settings_page(access_token: str):
    """Account settings page for logged-in users."""
    render_page_header(
        "Account Settings",
        "Manage your profile, writing identity, and account security in one place.",
        badges=["Profile", "Writer DNA", "Security"],
        compact=True,
    )

    try:
        response = requests.get(
            f"{API_BASE_URL}/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        if response.status_code != 200:
            st.error(_error_detail(response, "Failed to load profile"))
            return

        user = response.json()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Profile Information")
            st.text(f"Username: {user['username']}")
            st.text(f"Email: {user['email']}")
            email_status = "Verified" if user["email_verified"] else "Not verified"
            st.text(f"Email Status: {email_status}")

        with col2:
            st.subheader("Update Profile")
            bio = st.text_area("Bio", value=user.get("profile_bio") or "", max_chars=500)
            avatar_url = st.text_input("Avatar URL", value=user.get("avatar_url") or "")

            if st.button("Save Profile"):
                update_response = requests.put(
                    f"{API_BASE_URL}/users/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "profile_bio": bio,
                        "avatar_url": avatar_url,
                    },
                    timeout=10,
                )

                if update_response.status_code == 200:
                    st.success("Profile updated.")
                    st.rerun()
                else:
                    st.error(_error_detail(update_response, "Failed to update profile"))

        st.divider()

        if user.get("writer_profile"):
            st.subheader("Writer Profile")
            writer = user["writer_profile"]
            st.text(f"Style: {writer['writing_style']}")
            st.text(f"Topics: {', '.join(writer['primary_topics'])}")
            st.text(f"Submissions: {writer['total_submissions']}")
            if writer["avg_eqs_score"]:
                st.text(f"Avg Score: {writer['avg_eqs_score']:.1f}")

        st.divider()
        st.subheader("Security")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Change Password"):
                st.session_state.page = "forgot_password"
                st.rerun()

        with col2:
            if st.button("Verify Email"):
                st.session_state.pending_verification_email = user["email"]
                st.session_state.page = "verify_email"
                st.rerun()

        with col3:
            if st.button("Logout"):
                logout()

    except requests.exceptions.RequestException as exc:
        st.error(f"Connection error: {exc}")
