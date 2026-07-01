"""
API schemas for request/response validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, EmailStr, Field, validator


# ============================================================================
# AUTH SCHEMAS
# ============================================================================

class SignupRequest(BaseModel):
    """User registration request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain alphanumeric, - and _')
        return v


class LoginRequest(BaseModel):
    """User login request"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8)


class EmailVerificationRequest(BaseModel):
    """Email verification request"""
    token: str


class ResendVerificationRequest(BaseModel):
    """Resend email verification request"""
    email: EmailStr


# ============================================================================
# USER SCHEMAS
# ============================================================================

class WritingStyle(str, Enum):
    ANALYTICAL = "analytical"
    BALANCED = "balanced"
    EMOTIONAL = "emotional"
    DISMISSIVE = "dismissive"
    ATTACK_BASED = "attack_based"
    UNKNOWN = "unknown"


class WriterProfileResponse(BaseModel):
    """Writer profile response"""
    id: str
    user_id: str
    writing_style: WritingStyle
    primary_topics: List[str]
    total_submissions: int
    avg_eqs_score: Optional[float]
    reputation_points: int
    
    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    """User profile response"""
    id: str
    username: str
    email: str
    email_verified: bool
    profile_bio: Optional[str]
    avatar_url: Optional[str]
    created_at: datetime
    writer_profile: Optional[WriterProfileResponse]
    
    class Config:
        from_attributes = True


class UpdateUserRequest(BaseModel):
    """Update user profile"""
    profile_bio: Optional[str] = None
    avatar_url: Optional[str] = None


class UpdateWriterProfileRequest(BaseModel):
    """Update writer profile"""
    writing_style: Optional[WritingStyle] = None
    primary_topics: Optional[List[str]] = None


# ============================================================================
# COMMUNITY SCHEMAS
# ============================================================================

class CommunityTopic(str, Enum):
    BATTING = "batting"
    BOWLING = "bowling"
    FIELDING = "fielding"
    CAPTAINCY = "captaincy"
    STRATEGY = "strategy"
    TEAM_PERFORMANCE = "team_performance"
    PLAYER_COMPARISON = "player_comparison"
    TOURNAMENT = "tournament"
    GENERAL = "general"


class CommunityVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    INVITE_ONLY = "invite_only"


class CreateCommunityRequest(BaseModel):
    """Create new community"""
    name: str = Field(..., min_length=3, max_length=100)
    slug: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = None
    primary_topic: CommunityTopic
    secondary_topics: Optional[List[CommunityTopic]] = []
    visibility: CommunityVisibility = CommunityVisibility.PUBLIC
    
    @validator('slug')
    def slug_valid(cls, v):
        if not all(c.isalnum() or c in '-_' for c in v):
            raise ValueError('Slug can only contain alphanumeric, - and _')
        return v.lower()


class UpdateCommunityRequest(BaseModel):
    """Update community"""
    description: Optional[str] = None
    visibility: Optional[CommunityVisibility] = None
    rules: Optional[Dict[str, Any]] = None


class CommunityResponse(BaseModel):
    """Community response"""
    id: str
    name: str
    slug: str
    description: Optional[str]
    primary_topic: CommunityTopic
    secondary_topics: List[CommunityTopic]
    creator_id: str
    visibility: CommunityVisibility
    member_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class CommunityDetailResponse(CommunityResponse):
    """Detailed community response with members"""
    members: Optional[List[Dict[str, Any]]] = []


class JoinCommunityRequest(BaseModel):
    """Join community request"""
    pass  # Just need to verify user is authenticated


# ============================================================================
# SCORING SCHEMAS
# ============================================================================

class ScoringRequest(BaseModel):
    """Sentiment/quality scoring request"""
    text: str = Field(..., min_length=10, max_length=5000)
    community_id: Optional[str] = None
    model: str = Field(default="eqs", pattern="^(eqs|bqs|hybrid)$")


class ComponentScores(BaseModel):
    """Breakdown of scoring components"""
    stance_score: Optional[float] = None
    stance_label: Optional[str] = None
    stance_confidence: Optional[float] = None
    stats_verified: Optional[bool] = None
    writing_quality_score: Optional[float] = None
    toxicity_score: Optional[float] = None
    constructiveness_score: Optional[float] = None


class ScoringResponse(BaseModel):
    """Scoring result response"""
    id: str
    text: str
    model_version: str
    bqs_score: Optional[float]
    eqs_score: float
    confidence: float
    components: ComponentScores
    response_time_ms: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class ScoringHistoryResponse(BaseModel):
    """User scoring history"""
    total_submissions: int
    avg_eqs_score: float
    avg_confidence: float
    results: List[ScoringResponse]


# ============================================================================
# RATE LIMIT SCHEMAS
# ============================================================================

class RateLimitStatus(BaseModel):
    """Rate limit status"""
    model: str
    request_count: int
    max_requests: int
    usage_percentage: float
    status: str  # active, warning, blocked
    period_start: datetime
    period_end: datetime
    requests_remaining: int


# ============================================================================
# ERROR SCHEMAS
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
