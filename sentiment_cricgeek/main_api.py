"""
FastAPI backend for CricGeek Platform v2.0
Handles auth, scoring, communities, and user management
"""

import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Depends, Header, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from database import get_db, init_db
from auth_service import AuthService, ACCESS_TOKEN_EXPIRE_MINUTES
from eqs_service import EQSService
from community_service import CommunityService
from writer_data_service import WriterDataService
from models import User, ScoringResult
import schemas


def _enum_value(value):
    return getattr(value, "value", value)


# Initialize FastAPI app
app = FastAPI(
    title="CricGeek API v2.0",
    description="Enhanced sentiment scoring with communities and user management",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependency: Get current user from JWT token
# ============================================================================

async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate JWT token from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    
    # Verify token
    is_valid, payload = AuthService.verify_token(token)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before continuing",
        )
    
    return user


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
    }


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/signup", response_model=schemas.TokenResponse, tags=["Auth"])
async def signup(request: schemas.SignupRequest, db: Session = Depends(get_db)):
    """Register new user"""
    success, message, user = AuthService.register_user(
        username=request.username,
        email=request.email,
        password=request.password,
        db=db,
    )
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    # Generate tokens
    access_token = AuthService.create_access_token(user.id)
    refresh_token = AuthService.create_refresh_token(user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@app.post("/api/auth/login", response_model=schemas.TokenResponse, tags=["Auth"])
async def login(request: schemas.LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return tokens"""
    success, message, access_token = AuthService.login_user(
        username=request.username,
        password=request.password,
        db=db,
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=message,
        )
    
    # Get user to generate refresh token
    user = db.query(User).filter(
        (User.username == request.username) | (User.email == request.username)
    ).first()
    
    refresh_token = AuthService.create_refresh_token(user.id, db)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@app.post("/api/auth/refresh", response_model=schemas.TokenResponse, tags=["Auth"])
async def refresh_token(
    request: schemas.RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """Refresh access token using refresh token"""
    is_valid, user_id = AuthService.verify_refresh_token(request.refresh_token, db)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Generate new access token
    access_token = AuthService.create_access_token(user_id)
    
    return {
        "access_token": access_token,
        "refresh_token": request.refresh_token,  # Refresh token unchanged
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


@app.post("/api/auth/verify-email", tags=["Auth"])
async def verify_email(request: schemas.EmailVerificationRequest, db: Session = Depends(get_db)):
    """Verify email address"""
    success, message = AuthService.verify_email(request.token, db)
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    return {"message": message}


@app.post("/api/auth/resend-verification", tags=["Auth"])
async def resend_verification(request: schemas.ResendVerificationRequest, db: Session = Depends(get_db)):
    """Resend email verification instructions"""
    success, message = AuthService.resend_verification_email(request.email, db)

    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    return {"message": message}


@app.post("/api/auth/forgot-password", tags=["Auth"])
async def forgot_password(request: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    """Request password reset"""
    success, message = AuthService.request_password_reset(request.email, db)
    return {"message": message}


@app.post("/api/auth/reset-password", tags=["Auth"])
async def reset_password(request: schemas.PasswordResetConfirm, db: Session = Depends(get_db)):
    """Reset password with token"""
    success, message = AuthService.reset_password(request.token, request.new_password, db)
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    return {"message": message}


# ============================================================================
# USER ENDPOINTS
# ============================================================================

@app.get("/api/users/me", response_model=schemas.UserResponse, tags=["Users"])
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Get current user profile"""
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "email_verified": current_user.email_verified,
        "profile_bio": current_user.profile_bio,
        "avatar_url": current_user.avatar_url,
        "created_at": current_user.created_at,
        "writer_profile": current_user.writer_profile,
    }


@app.put("/api/users/me", response_model=schemas.UserResponse, tags=["Users"])
async def update_user_profile(
    request: schemas.UpdateUserRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user profile"""
    if request.profile_bio is not None:
        current_user.profile_bio = request.profile_bio
    if request.avatar_url is not None:
        current_user.avatar_url = request.avatar_url
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "email_verified": current_user.email_verified,
        "profile_bio": current_user.profile_bio,
        "avatar_url": current_user.avatar_url,
        "created_at": current_user.created_at,
        "writer_profile": current_user.writer_profile,
    }


@app.put("/api/users/me/writer-profile", response_model=schemas.WriterProfileResponse, tags=["Users"])
async def update_writer_profile(
    request: schemas.UpdateWriterProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update writer profile"""
    from models import WriterProfile
    
    if not current_user.writer_profile:
        # Create if doesn't exist
        from uuid import uuid4
        writer_profile = WriterProfile(
            user_id=current_user.id,
        )
        db.add(writer_profile)
        db.flush()
    else:
        writer_profile = current_user.writer_profile
    
    if request.writing_style:
        writer_profile.writing_style = _enum_value(request.writing_style)
    if request.primary_topics:
        writer_profile.primary_topics = [_enum_value(topic) for topic in request.primary_topics]
    
    writer_profile.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(writer_profile)
    
    return {
        "id": str(writer_profile.id),
        "user_id": str(writer_profile.user_id),
        "writing_style": _enum_value(writer_profile.writing_style),
        "primary_topics": writer_profile.primary_topics,
        "total_submissions": writer_profile.total_submissions,
        "avg_eqs_score": writer_profile.avg_eqs_score,
        "reputation_points": writer_profile.reputation_points,
    }


# ============================================================================
# SCORING ENDPOINTS
# ============================================================================

@app.post("/api/score", response_model=schemas.ScoringResponse, tags=["Scoring"])
async def score_text(
    request: schemas.ScoringRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Score cricket commentary text"""
    # Check rate limit
    if request.model in ["eqs", "hybrid"]:
        allowed, limit_msg = EQSService.check_rate_limit(str(current_user.id), db)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=limit_msg,
            )
    
    # Score the text
    scoring_result = EQSService.score_text(
        text=request.text,
        user_id=str(current_user.id),
        db=db,
        community_id=request.community_id,
        use_legacy_bqs=(request.model == "bqs"),
    )
    
    if not scoring_result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=scoring_result.get("error", "Scoring failed"),
        )
    
    try:
        result_obj = WriterDataService.record_scoring_result(
            user_id=current_user.id,
            text=request.text,
            scoring_result=scoring_result,
            community_id=request.community_id,
            db=db,
        )
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")
    
    return {
        "id": str(result_obj.id),
        "text": result_obj.text,
        "model_version": result_obj.model_version,
        "bqs_score": result_obj.bqs_score,
        "eqs_score": result_obj.eqs_score,
        "confidence": result_obj.confidence,
        "components": result_obj.components,
        "response_time_ms": result_obj.response_time_ms,
        "created_at": result_obj.created_at,
    }


@app.get("/api/score/history", tags=["Scoring"])
async def get_scoring_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's scoring history"""
    results = db.query(ScoringResult).filter(
        ScoringResult.user_id == current_user.id
    ).order_by(ScoringResult.created_at.desc()).offset(offset).limit(limit).all()
    
    total = db.query(ScoringResult).filter(
        ScoringResult.user_id == current_user.id
    ).count()
    
    return {
        "total": total,
        "results": [
            {
                "id": str(r.id),
                "text": r.text,
                "model_version": r.model_version,
                "eqs_score": r.eqs_score,
                "confidence": r.confidence,
                "created_at": r.created_at,
            }
            for r in results
        ]
    }


@app.get("/api/rate-limit/status", tags=["Scoring"])
async def get_rate_limit_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get current rate limit status for Qwen 3.5"""
    status_info = EQSService.get_rate_limit_status(str(current_user.id), db)
    return status_info


# ============================================================================
# COMMUNITY ENDPOINTS
# ============================================================================

@app.post("/api/communities", response_model=schemas.CommunityResponse, tags=["Communities"])
async def create_community(
    request: schemas.CreateCommunityRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create new community"""
    success, message, community = CommunityService.create_community(
        name=request.name,
        slug=request.slug,
        description=request.description,
        primary_topic=_enum_value(request.primary_topic),
        secondary_topics=[_enum_value(topic) for topic in (request.secondary_topics or [])],
        creator_id=current_user.id,
        visibility=_enum_value(request.visibility),
        db=db,
    )
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    return {
        "id": str(community.id),
        "name": community.name,
        "slug": community.slug,
        "description": community.description,
        "primary_topic": _enum_value(community.primary_topic),
        "secondary_topics": community.secondary_topics,
        "creator_id": str(community.creator_id),
        "visibility": _enum_value(community.visibility),
        "member_count": community.member_count,
        "created_at": community.created_at,
    }


@app.get("/api/communities", tags=["Communities"])
async def list_communities(
    topic: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List communities with filters"""
    total, communities = CommunityService.list_communities(
        topic=topic,
        visibility="public",
        search=search,
        limit=limit,
        offset=offset,
        db=db,
    )
    
    return {
        "total": total,
        "communities": [
            {
                "id": str(c.id),
                "name": c.name,
                "slug": c.slug,
                "description": c.description,
                "primary_topic": _enum_value(c.primary_topic),
                "member_count": c.member_count,
                "created_at": c.created_at,
            }
            for c in communities
        ]
    }


@app.get("/api/communities/discover", tags=["Communities"])
async def discover_communities(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Discover communities for current user"""
    communities = CommunityService.discover_communities_for_writer(
        current_user.id,
        limit=10,
        db=db,
    )
    
    return {
        "communities": [
            {
                "id": str(c.id),
                "name": c.name,
                "slug": c.slug,
                "description": c.description,
                "primary_topic": _enum_value(c.primary_topic),
                "member_count": c.member_count,
            }
            for c in communities
        ]
    }


@app.get("/api/writers/same-topic", tags=["Communities"])
async def discover_same_topic_writers(
    topic: Optional[str] = Query(None),
    writing_style: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Discover like-minded writers by shared topic and optional writing style"""
    writers = CommunityService.find_like_minded_writers(
        user_id=current_user.id,
        topic=topic,
        writing_style=writing_style,
        limit=limit,
        db=db,
    )
    return {"writers": writers}


@app.get("/api/communities/{community_id}/same-topic-writers", tags=["Communities"])
async def discover_community_writers(
    community_id: str,
    writing_style: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Discover writers who match a community's primary topic"""
    from uuid import UUID
    try:
        comm_id = UUID(community_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")

    community = CommunityService.get_community(comm_id, db)
    if not community:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Community not found")

    writers = CommunityService.find_like_minded_writers(
        user_id=current_user.id,
        topic=community.primary_topic,
        writing_style=writing_style,
        limit=limit,
        db=db,
    )
    return {"writers": writers}


@app.get("/api/communities/{community_id}", tags=["Communities"])
async def get_community(
    community_id: str,
    db: Session = Depends(get_db),
):
    """Get community details"""
    from uuid import UUID
    try:
        comm_id = UUID(community_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")
    
    community = CommunityService.get_community(comm_id, db)
    
    if not community:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Community not found")
    
    members = CommunityService.get_community_members(comm_id, limit=5, db=db)
    
    return {
        "id": str(community.id),
        "name": community.name,
        "slug": community.slug,
        "description": community.description,
        "primary_topic": _enum_value(community.primary_topic),
        "creator_id": str(community.creator_id),
        "visibility": _enum_value(community.visibility),
        "member_count": community.member_count,
        "members": members,
        "created_at": community.created_at,
    }


@app.post("/api/communities/{community_id}/join", tags=["Communities"])
async def join_community(
    community_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Join a community"""
    from uuid import UUID
    try:
        comm_id = UUID(community_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")
    
    success, message = CommunityService.join_community(comm_id, current_user.id, db)
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    return {"message": message}


@app.post("/api/communities/{community_id}/leave", tags=["Communities"])
async def leave_community(
    community_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Leave a community"""
    from uuid import UUID
    try:
        comm_id = UUID(community_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")
    
    success, message = CommunityService.leave_community(comm_id, current_user.id, db)
    
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    return {"message": message}


@app.get("/api/communities/{community_id}/members", tags=["Communities"])
async def get_community_members(
    community_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get community members"""
    from uuid import UUID
    try:
        comm_id = UUID(community_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid community ID")
    
    members = CommunityService.get_community_members(comm_id, limit, offset, db)
    
    return {"members": members}


@app.get("/api/users/me/communities", tags=["Communities"])
async def get_my_communities(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get communities current user is member of"""
    communities = CommunityService.get_user_communities(
        current_user.id,
        limit=50,
        db=db,
    )
    
    return {
        "communities": [
            {
                "id": str(c.id),
                "name": c.name,
                "slug": c.slug,
                "primary_topic": _enum_value(c.primary_topic),
                "member_count": c.member_count,
            }
            for c in communities
        ]
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        print("[✓] Database initialized")
    except Exception as e:
        print(f"[!] Database initialization error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
