"""
Database models for CricGeek Platform v2.0
- User authentication and profiles
- Communities and memberships
- Scoring history and rate limiting
- Encrypted sensitive data
"""

import base64
import hashlib
import os
from datetime import datetime, timedelta
from enum import Enum
from uuid import uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum, Float, ForeignKey,
    Integer, JSON, String, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import CHAR, TypeDecorator
from cryptography.fernet import Fernet
import bcrypt

Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent UUID type stored as a 36-character string."""

    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return str(value)

# Encryption setup
class EncryptedString(TypeDecorator):
    """Encrypt string values before writing them to the database."""
    impl = Text
    cache_ok = True

    @classmethod
    def _get_cipher(cls):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            if os.getenv("ENVIRONMENT", "development").lower() == "production":
                raise RuntimeError("ENCRYPTION_KEY is required in production")
            # Development-only fallback so local SQLite setup still works.
            key = base64.urlsafe_b64encode(b"cricgeek-local-dev-key-000000000")
        return Fernet(key)

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        cipher = self._get_cipher()
        encrypted = cipher.encrypt(value.encode())
        return encrypted.decode()

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        cipher = self._get_cipher()
        decrypted = cipher.decrypt(value.encode())
        return decrypted.decode()


def hash_lookup_token(token: str) -> str:
    """One-way hash for verification/reset token lookup."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class WritingStyleEnum(str, Enum):
    """Writer's primary communication style"""
    ANALYTICAL = "analytical"
    BALANCED = "balanced"
    EMOTIONAL = "emotional"
    DISMISSIVE = "dismissive"
    ATTACK_BASED = "attack_based"
    UNKNOWN = "unknown"


class TopicEnum(str, Enum):
    """Cricket discussion topics"""
    BATTING = "batting"
    BOWLING = "bowling"
    FIELDING = "fielding"
    CAPTAINCY = "captaincy"
    STRATEGY = "strategy"
    TEAM_PERFORMANCE = "team_performance"
    PLAYER_COMPARISON = "player_comparison"
    TOURNAMENT = "tournament"
    GENERAL = "general"


class CommunityVisibilityEnum(str, Enum):
    """Community visibility level"""
    PUBLIC = "public"
    PRIVATE = "private"
    INVITE_ONLY = "invite_only"


class MemberRoleEnum(str, Enum):
    """Role in a community"""
    OWNER = "owner"
    MODERATOR = "moderator"
    MEMBER = "member"


class RateLimitStatusEnum(str, Enum):
    """Rate limit status"""
    ACTIVE = "active"
    WARNING = "warning"
    BLOCKED = "blocked"


class User(Base):
    """User account and authentication"""
    __tablename__ = "users"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Email verification
    email_verified = Column(Boolean, default=False)
    verification_token_hash = Column("verification_token", String(64), nullable=True, index=True)
    verification_token_expires = Column(DateTime, nullable=True)
    
    # Profile
    profile_bio = Column(Text, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # Account management
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    writer_profile = relationship("WriterProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    owned_communities = relationship("Community", back_populates="creator", foreign_keys="Community.creator_id")
    community_memberships = relationship("CommunityMember", back_populates="user", cascade="all, delete-orphan")
    scoring_results = relationship("ScoringResult", back_populates="user", cascade="all, delete-orphan")
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    rate_limit_logs = relationship("RateLimitLog", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_created', 'created_at'),
    )
    
    @staticmethod
    def hash_password(password: str, rounds: int = None) -> str:
        """Hash password using bcrypt"""
        if rounds is None:
            rounds = int(os.getenv("BCRYPT_ROUNDS", "12"))
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds)).decode()
    
    def verify_password(self, password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), self.password_hash.encode())


class RefreshToken(Base):
    """JWT refresh token tracker (for token revocation)"""
    __tablename__ = "refresh_tokens"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, ForeignKey('users.id'), nullable=False, index=True)
    token_jti = Column(String(255), unique=True, nullable=False)  # JWT ID claim
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="refresh_tokens")
    
    __table_args__ = (
        Index('idx_refresh_token_user_expires', 'user_id', 'expires_at'),
    )


class WriterProfile(Base):
    """Extended profile for writers"""
    __tablename__ = "writer_profiles"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, ForeignKey('users.id'), unique=True, nullable=False, index=True)
    
    # Writing characteristics
    writing_style = Column(SQLEnum(WritingStyleEnum), default=WritingStyleEnum.UNKNOWN)
    primary_topics = Column(JSON, default=list)  # List of TopicEnum
    
    # Statistics
    total_submissions = Column(Integer, default=0)
    avg_eqs_score = Column(Float, nullable=True)
    avg_bqs_score = Column(Float, nullable=True)  # Legacy
    reputation_points = Column(Integer, default=0)
    
    # Profile metadata
    profile_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="writer_profile")


class Community(Base):
    """Community of like-minded writers"""
    __tablename__ = "communities"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Topic focus
    primary_topic = Column(SQLEnum(TopicEnum), nullable=False, index=True)
    secondary_topics = Column(JSON, default=list)
    
    # Management
    creator_id = Column(GUID, ForeignKey('users.id'), nullable=False, index=True)
    visibility = Column(SQLEnum(CommunityVisibilityEnum), default=CommunityVisibilityEnum.PUBLIC)
    
    # Stats
    member_count = Column(Integer, default=0)
    rules = Column(JSON, default=dict)
    
    # Metadata
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    creator = relationship("User", back_populates="owned_communities", foreign_keys=[creator_id])
    members = relationship("CommunityMember", back_populates="community", cascade="all, delete-orphan")
    scoring_results = relationship("ScoringResult", back_populates="community")
    
    __table_args__ = (
        Index('idx_community_topic_visibility', 'primary_topic', 'visibility'),
        Index('idx_community_created', 'created_at'),
    )


class CommunityMember(Base):
    """Community membership tracking"""
    __tablename__ = "community_members"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    community_id = Column(GUID, ForeignKey('communities.id'), nullable=False, index=True)
    user_id = Column(GUID, ForeignKey('users.id'), nullable=False, index=True)
    
    role = Column(SQLEnum(MemberRoleEnum), default=MemberRoleEnum.MEMBER)
    joined_at = Column(DateTime, default=datetime.utcnow)
    last_active_at = Column(DateTime, nullable=True)
    
    # Relationships
    community = relationship("Community", back_populates="members")
    user = relationship("User", back_populates="community_memberships")
    
    __table_args__ = (
        UniqueConstraint('community_id', 'user_id', name='uq_community_user'),
        Index('idx_community_member_user', 'user_id'),
    )


class ScoringResult(Base):
    """Sentiment/quality scoring result"""
    __tablename__ = "scoring_results"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, ForeignKey('users.id'), nullable=False, index=True)
    community_id = Column(GUID, ForeignKey('communities.id'), nullable=True, index=True)
    
    # Text and model info
    text = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)
    model_version = Column(String(50), nullable=False)  # e.g., "EQS-Qwen3.5-v1"
    
    # Scores (legacy BQS + new EQS)
    bqs_score = Column(Float, nullable=True)  # Big Query Service (legacy)
    eqs_score = Column(Float, nullable=False)  # Enhanced Query Service (Qwen 3.5)
    confidence = Column(Float, nullable=False)
    
    # Component breakdown (JSON)
    components = Column(JSON, nullable=False)  # {stance, stats_verified, writing_quality, etc.}
    
    # Metadata
    request_timestamp = Column(DateTime, default=datetime.utcnow, index=True)  # For rate limiting
    response_time_ms = Column(Integer, nullable=True)
    is_flagged = Column(Boolean, default=False)
    flag_reason = Column(String(255), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="scoring_results")
    community = relationship("Community", back_populates="scoring_results")
    
    __table_args__ = (
        Index('idx_scoring_user_created', 'user_id', 'created_at'),
        Index('idx_scoring_community_created', 'community_id', 'created_at'),
        Index('idx_scoring_request_timestamp', 'request_timestamp'),
    )


class RateLimitLog(Base):
    """Track rate limit usage for Qwen 3.5"""
    __tablename__ = "rate_limit_logs"
    
    id = Column(GUID, primary_key=True, default=uuid4)
    user_id = Column(GUID, ForeignKey('users.id'), nullable=False, index=True)
    
    model = Column(String(50), nullable=False)  # 'qwen3.5'
    request_count = Column(Integer, default=0)
    
    # Period (e.g., 30 days)
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Status
    status = Column(SQLEnum(RateLimitStatusEnum), default=RateLimitStatusEnum.ACTIVE)
    
    # Limits
    max_requests = Column(Integer, default=1000)
    warning_threshold = Column(Float, default=0.8)  # Warn at 80%
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="rate_limit_logs")
    
    __table_args__ = (
        Index('idx_rate_limit_user_period', 'user_id', 'period_start'),
    )
    
    def is_over_limit(self) -> bool:
        """Check if user has exceeded rate limit"""
        return self.request_count >= self.max_requests
    
    def get_usage_percentage(self) -> float:
        """Get usage as percentage"""
        if not self.max_requests:
            return 0.0
        return (self.request_count / self.max_requests) * 100

    def should_warn(self) -> bool:
        """Return True when usage is past the warning threshold but not blocked."""
        if not self.max_requests:
            return False
        return self.request_count >= int(self.max_requests * self.warning_threshold)
    
    def should_warn(self) -> bool:
        """Check if should show warning"""
        return self.get_usage_percentage() >= (self.warning_threshold * 100)
