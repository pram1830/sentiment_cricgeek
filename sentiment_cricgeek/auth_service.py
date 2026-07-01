"""
Authentication service with JWT, email verification, and password management
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import jwt
from sqlalchemy.orm import Session

from models import User, RefreshToken, hash_lookup_token
from database import SessionLocal


# JWT configuration
def _load_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    environment = os.getenv("ENVIRONMENT", "development").lower()
    if not secret:
        if environment == "production":
            raise RuntimeError("JWT_SECRET_KEY is required in production")
        return "cricgeek-local-development-jwt-secret-change-me"
    if environment == "production" and len(secret) < 32:
        raise RuntimeError("JWT_SECRET_KEY must be at least 32 characters in production")
    return secret


JWT_SECRET_KEY = _load_jwt_secret()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Email configuration
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "console")  # console, sendgrid, mailgun
EMAIL_API_KEY = os.getenv("EMAIL_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "noreply@cricgeek.local")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8501")


def _should_require_email_verification() -> bool:
    """Require email verification in production and when explicitly requested."""
    explicit_setting = os.getenv("REQUIRE_EMAIL_VERIFICATION", "").strip().lower()
    if explicit_setting in {"1", "true", "yes", "on"}:
        return True
    if explicit_setting in {"0", "false", "no", "off"}:
        return False

    environment = os.getenv("ENVIRONMENT", "development").lower()
    if environment == "production":
        return True

    return EMAIL_PROVIDER != "console"


class AuthService:
    """Handle user authentication, JWT tokens, and email verification"""
    
    @staticmethod
    def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            user_id: UUID of user
            expires_delta: Custom expiration time
            
        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        expire = datetime.utcnow() + expires_delta
        payload = {
            "sub": str(user_id),
            "type": "access",
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return token
    
    @staticmethod
    def create_refresh_token(user_id: str, db: Session) -> str:
        """
        Create JWT refresh token and store in database
        
        Args:
            user_id: UUID of user
            db: Database session
            
        Returns:
            JWT token string
        """
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        expire = datetime.utcnow() + expires_delta
        jti = secrets.token_urlsafe(32)
        
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti,
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # Store in database for revocation support
        refresh_token_record = RefreshToken(
            user_id=user_id,
            token_jti=jti,
            expires_at=expire,
        )
        db.add(refresh_token_record)
        db.commit()
        
        return token
    
    @staticmethod
    def verify_token(token: str) -> Tuple[bool, Optional[dict]]:
        """
        Verify JWT token and return payload
        
        Args:
            token: JWT token string
            
        Returns:
            Tuple of (is_valid, payload)
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}
    
    @staticmethod
    def verify_refresh_token(token: str, db: Session) -> Tuple[bool, Optional[str]]:
        """
        Verify refresh token and check if revoked
        
        Args:
            token: JWT token string
            db: Database session
            
        Returns:
            Tuple of (is_valid, user_id)
        """
        is_valid, payload = AuthService.verify_token(token)
        
        if not is_valid or payload.get("type") != "refresh":
            return False, None
        
        user_id = payload.get("sub")
        jti = payload.get("jti")
        
        # Check if token is revoked
        refresh_token = db.query(RefreshToken).filter(
            RefreshToken.token_jti == jti,
            RefreshToken.revoked == False,
        ).first()
        
        if not refresh_token:
            return False, None
        
        return True, user_id
    
    @staticmethod
    def register_user(username: str, email: str, password: str, db: Session) -> Tuple[bool, str, Optional[User]]:
        """
        Register new user
        
        Args:
            username: Username (50 chars max)
            email: Email address
            password: Plain password (will be hashed)
            db: Database session
            
        Returns:
            Tuple of (success, message, user_object)
        """
        normalized_email = email.strip().lower()
        normalized_username = username.strip()

        # Validate input
        if len(normalized_username) < 3 or len(normalized_username) > 50:
            return False, "Username must be 3-50 characters", None
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters", None
        
        if "@" not in normalized_email or len(normalized_email) < 5:
            return False, "Invalid email address", None
        
        # Check if username exists
        if db.query(User).filter(User.username == normalized_username).first():
            return False, "Username already exists", None
        
        # Check if email exists
        if db.query(User).filter(User.email == normalized_email).first():
            return False, "Email already registered", None
        
        # Create user
        try:
            user = User(
                username=normalized_username,
                email=normalized_email,
                password_hash=User.hash_password(password),
                email_verified=not _should_require_email_verification(),
            )
            
            # Generate verification token
            verification_token = AuthService.generate_verification_token()
            user.verification_token_hash = hash_lookup_token(verification_token)
            user.verification_token_expires = datetime.utcnow() + timedelta(hours=24)
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Send verification email
            AuthService.send_verification_email(normalized_email, verification_token)
            
            return True, "User registered. Check email for verification link", user
        
        except Exception as e:
            db.rollback()
            return False, f"Registration failed: {str(e)}", None
    
    @staticmethod
    def login_user(username: str, password: str, db: Session) -> Tuple[bool, str, Optional[str]]:
        """
        Authenticate user and return access token
        
        Args:
            username: Username or email
            password: Plain password
            db: Database session
            
        Returns:
            Tuple of (success, message, access_token)
        """
        # Find user by username or email
        login_id = username.strip()
        login_email = login_id.lower()
        user = db.query(User).filter(
            (User.username == login_id) | (User.email == login_email)
        ).first()
        
        if not user:
            return False, "Invalid username or password", None
        
        if not user.is_active:
            return False, "Account is disabled", None
        
        if not user.verify_password(password):
            return False, "Invalid username or password", None
        
        if _should_require_email_verification() and not user.email_verified:
            return False, "Please verify your email before logging in", None
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Generate tokens
        access_token = AuthService.create_access_token(user.id)
        
        return True, "Login successful", access_token
    
    @staticmethod
    def verify_email(token: str, db: Session) -> Tuple[bool, str]:
        """
        Verify email with verification token
        
        Args:
            token: Verification token
            db: Database session
            
        Returns:
            Tuple of (success, message)
        """
        token_hash = hash_lookup_token(token)
        user = db.query(User).filter(
            User.verification_token_hash == token_hash,
            User.verification_token_expires > datetime.utcnow(),
        ).first()
        
        if not user:
            return False, "Invalid or expired verification token"
        
        user.email_verified = True
        user.verification_token_hash = None
        user.verification_token_expires = None
        db.commit()
        
        return True, "Email verified successfully"

    @staticmethod
    def resend_verification_email(email: str, db: Session) -> Tuple[bool, str]:
        """
        Resend an email verification token.

        The response is intentionally generic so callers cannot use this
        endpoint to discover registered emails.
        """
        normalized_email = email.strip().lower()
        user = db.query(User).filter(User.email == normalized_email).first()

        generic_message = "If the account exists and is unverified, a verification email has been sent"
        if not user or user.email_verified:
            return True, generic_message

        verification_token = AuthService.generate_verification_token()
        user.verification_token_hash = hash_lookup_token(verification_token)
        user.verification_token_expires = datetime.utcnow() + timedelta(hours=24)
        db.commit()

        AuthService.send_verification_email(normalized_email, verification_token)
        return True, generic_message
    
    @staticmethod
    def request_password_reset(email: str, db: Session) -> Tuple[bool, str]:
        """
        Request password reset
        
        Args:
            email: User email
            db: Database session
            
        Returns:
            Tuple of (success, message)
        """
        normalized_email = email.strip().lower()
        user = db.query(User).filter(User.email == normalized_email).first()
        
        if not user:
            # Don't reveal if email exists (security)
            return True, "If email exists, reset link has been sent"
        
        # Generate reset token (reuse verification_token field)
        reset_token = AuthService.generate_verification_token()
        user.verification_token_hash = hash_lookup_token(reset_token)
        user.verification_token_expires = datetime.utcnow() + timedelta(hours=1)
        db.commit()
        
        # Send reset email
        AuthService.send_password_reset_email(normalized_email, reset_token)
        
        return True, "If email exists, reset link has been sent"
    
    @staticmethod
    def reset_password(token: str, new_password: str, db: Session) -> Tuple[bool, str]:
        """
        Reset password with token
        
        Args:
            token: Password reset token
            new_password: New password
            db: Database session
            
        Returns:
            Tuple of (success, message)
        """
        if len(new_password) < 8:
            return False, "Password must be at least 8 characters"
        
        token_hash = hash_lookup_token(token)
        user = db.query(User).filter(
            User.verification_token_hash == token_hash,
            User.verification_token_expires > datetime.utcnow(),
        ).first()
        
        if not user:
            return False, "Invalid or expired reset token"
        
        user.password_hash = User.hash_password(new_password)
        user.verification_token_hash = None
        user.verification_token_expires = None
        db.commit()
        
        return True, "Password reset successful"
    
    @staticmethod
    def generate_verification_token() -> str:
        """Generate random verification token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def send_verification_email(email: str, token: str):
        """Send verification email"""
        verification_url = f"{APP_BASE_URL}/?page=verify_email&token={token}"
        if EMAIL_PROVIDER == "console":
            print(f"\n[EMAIL] Verification link for {email}: {verification_url}\n")
        elif EMAIL_PROVIDER == "sendgrid":
            AuthService._send_via_sendgrid(
                email,
                "Verify your CricGeek email",
                f"Click here to verify: {verification_url}"
            )
        elif EMAIL_PROVIDER == "mailgun":
            AuthService._send_via_mailgun(
                email,
                "Verify your CricGeek email",
                f"Click here to verify: {verification_url}"
            )
    
    @staticmethod
    def send_password_reset_email(email: str, token: str):
        """Send password reset email"""
        reset_url = f"{APP_BASE_URL}/?page=reset_password&token={token}"
        if EMAIL_PROVIDER == "console":
            print(f"\n[EMAIL] Password reset link for {email}: {reset_url}\n")
        elif EMAIL_PROVIDER == "sendgrid":
            AuthService._send_via_sendgrid(
                email,
                "Reset your CricGeek password",
                f"Click here to reset: {reset_url}"
            )
        elif EMAIL_PROVIDER == "mailgun":
            AuthService._send_via_mailgun(
                email,
                "Reset your CricGeek password",
                f"Click here to reset: {reset_url}"
            )
    
    @staticmethod
    def _send_via_sendgrid(to_email: str, subject: str, body: str):
        """Send email via SendGrid"""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            
            message = Mail(
                from_email=SENDER_EMAIL,
                to_emails=to_email,
                subject=subject,
                plain_text_content=body,
            )
            
            sg = SendGridAPIClient(EMAIL_API_KEY)
            sg.send(message)
        except Exception as e:
            print(f"Failed to send email via SendGrid: {e}")
    
    @staticmethod
    def _send_via_mailgun(to_email: str, subject: str, body: str):
        """Send email via Mailgun"""
        try:
            import requests
            
            domain = EMAIL_API_KEY.split("mg.")[1].split(":")[0] if "mg." in EMAIL_API_KEY else "mg.cricgeek.local"
            
            requests.post(
                f"https://api.mailgun.net/v3/{domain}/messages",
                auth=("api", EMAIL_API_KEY),
                data={
                    "from": SENDER_EMAIL,
                    "to": to_email,
                    "subject": subject,
                    "text": body,
                }
            )
        except Exception as e:
            print(f"Failed to send email via Mailgun: {e}")
