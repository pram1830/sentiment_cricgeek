import os
import time
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


class ApiTestError(RuntimeError):
    pass


def assert_status(resp, expected, context):
    if resp.status_code != expected:
        raise ApiTestError(f"{context}: expected {expected}, got {resp.status_code}: {resp.text}")
    return resp


def assert_json(resp, context):
    try:
        return resp.json()
    except ValueError as exc:
        raise ApiTestError(f"{context}: response is not valid JSON: {resp.text}") from exc


def main():
    print("Testing health endpoint...")
    health = assert_status(requests.get(f"{BASE_URL}/health"), 200, "health")
    assert_json(health, "health")

    print("Testing auth signup...")
    unique = int(time.time())
    signup_payload = {
        "username": f"smoketest{unique}",
        "email": f"smoketest{unique}@example.com",
        "password": "TestPassword123!",
    }
    signup = requests.post(f"{BASE_URL}/api/auth/signup", json=signup_payload)
    assert_status(signup, 200, "signup")
    signup_body = assert_json(signup, "signup")
    assert "access_token" in signup_body and "refresh_token" in signup_body

    print("Testing auth login...")
    login = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": signup_payload["username"], "password": signup_payload["password"]},
    )
    assert_status(login, 200, "login")
    login_body = assert_json(login, "login")
    assert "access_token" in login_body and "refresh_token" in login_body

    access_token = login_body["access_token"]
    refresh_token = login_body["refresh_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    print("Testing refresh token...")
    refresh = requests.post(
        f"{BASE_URL}/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert_status(refresh, 200, "refresh")
    assert_json(refresh, "refresh")

    print("Testing verify-email endpoint...")
    verify = requests.post(f"{BASE_URL}/api/auth/verify-email", json={"token": "invalid-token"})
    assert_status(verify, 400, "verify-email")
    assert_json(verify, "verify-email")

    print("Testing resend-verification endpoint...")
    resend = requests.post(f"{BASE_URL}/api/auth/resend-verification", json={"email": signup_payload["email"]})
    assert_status(resend, 200, "resend-verification")
    assert_json(resend, "resend-verification")

    print("Testing forgot-password endpoint...")
    forgot = requests.post(f"{BASE_URL}/api/auth/forgot-password", json={"email": signup_payload["email"]})
    assert_status(forgot, 200, "forgot-password")
    assert_json(forgot, "forgot-password")

    print("Testing protected user endpoint...")
    me = requests.get(f"{BASE_URL}/api/users/me", headers=headers)
    assert_status(me, 200, "users/me")
    assert_json(me, "users/me")

    print("Testing scoring history endpoint...")
    history = requests.get(f"{BASE_URL}/api/score/history", headers=headers)
    assert_status(history, 200, "score/history")
    assert_json(history, "score/history")

    print("Testing rate-limit endpoint...")
    rate_limit = requests.get(f"{BASE_URL}/api/rate-limit/status", headers=headers)
    assert_status(rate_limit, 200, "rate-limit/status")
    assert_json(rate_limit, "rate-limit/status")

    print("Testing community create/list...")
    community_payload = {
        "name": f"Smoke Community {unique}",
        "slug": f"smoke-community-{unique}",
        "description": "Community created by smoke test",
        "primary_topic": "strategy",
        "secondary_topics": ["batting"],
        "visibility": "public",
    }
    created_community = requests.post(f"{BASE_URL}/api/communities", json=community_payload, headers=headers)
    assert_status(created_community, 200, "create-community")
    community_body = assert_json(created_community, "create-community")
    community_id = community_body["id"]

    communities = requests.get(f"{BASE_URL}/api/communities")
    assert_status(communities, 200, "list-communities")
    assert_json(communities, "list-communities")

    print("Testing community detail and membership...")
    detail = requests.get(f"{BASE_URL}/api/communities/{community_id}", headers=headers)
    assert_status(detail, 200, "community-detail")
    assert_json(detail, "community-detail")

    join = requests.post(f"{BASE_URL}/api/communities/{community_id}/join", headers=headers)
    assert_status(join, 200, "join-community")
    assert_json(join, "join-community")

    joined = requests.get(f"{BASE_URL}/api/users/me/communities", headers=headers)
    assert_status(joined, 200, "my-communities")
    assert_json(joined, "my-communities")

    leave = requests.post(f"{BASE_URL}/api/communities/{community_id}/leave", headers=headers)
    assert_status(leave, 200, "leave-community")
    assert_json(leave, "leave-community")

    print("Testing scoring endpoint...")
    score = requests.post(
        f"{BASE_URL}/api/score",
        json={
            "text": "India won the final over with smart bowling and calm decision-making.",
            "model": "eqs",
            "community_id": community_id,
        },
        headers=headers,
    )
    assert_status(score, 200, "score")
    score_body = assert_json(score, "score")
    assert "id" in score_body and "eqs_score" in score_body

    print("Testing database records were created...")
    from database import SessionLocal
    from models import User, ScoringResult, Community, RefreshToken

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == signup_payload["username"]).first()
        if not user:
            raise ApiTestError("user not persisted")
        if not db.query(RefreshToken).filter(RefreshToken.user_id == user.id).count():
            raise ApiTestError("refresh token not persisted")
        if not db.query(Community).filter(Community.slug == community_payload["slug"]).count():
            raise ApiTestError("community not persisted")
        if not db.query(ScoringResult).filter(ScoringResult.user_id == user.id).count():
            raise ApiTestError("scoring result not persisted")
    finally:
        db.close()

    print("All API smoke tests passed.")


if __name__ == "__main__":
    main()
