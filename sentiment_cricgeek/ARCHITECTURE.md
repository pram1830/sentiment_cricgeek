# CricGeek Platform Architecture v2.0

## Overview
Migration from BQS (Big Query Service) to EQS (Enhanced Query Service) with multi-tenant support, community features, and secure user management.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                      │
│  ┌──────────────┬──────────────┬──────────────────────┐ │
│  │  Auth Pages  │  Dashboard   │  Community Browser   │ │
│  │ (Login/Sign) │  (Scoring)   │  (Discovery)         │ │
│  └──────┬───────┴──────┬───────┴────────┬─────────────┘ │
└─────────┼──────────────┼────────────────┼────────────────┘
          │              │                │
┌─────────┼──────────────┼────────────────┼────────────────┐
│         │ Session Mgmt │ Scoring Req    │ Community Mgmt │
│    AuthService    │  SentimentPipeline  │  CommunityMgr  │
│         │         │                     │                │
└─────────┼─────────┼─────────────────────┼────────────────┘
          │         │                     │
┌─────────────────────────────────────────────────────────┐
│              API Layer (FastAPI)                         │
│  ┌──────────┬──────────┬──────────┬──────────────────┐  │
│  │ Auth     │ Scoring  │ Users    │ Communities      │  │
│  │ Endpoints│ Endpoints│ Endpoints│ Endpoints        │  │
│  └──────────┴──────────┴──────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │
┌─────────┴──────────────────────────────────────────────┐
│  │  PostgreSQL / MySQL Database                     │  │
│  ┌──────────────┬──────────────┬────────────────────┐  │
│  │ AuthManager  │ SentimentMgr  │ CommunityManager  │  │
│  │ (JWT, Email) │ (EQS+Qwen 3.5)│ (Topics, Writers) │  │
│  └──────────────┴──────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │
┌─────────┴──────────────────────────────────────────────┐
│           Data Layer (SQLAlchemy ORM)                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  PostgreSQL Database                             │  │
│  │  • Users & Auth Tokens                           │  │
│  │  • Writer Profiles                               │  │
│  │  • Communities & Memberships                     │  │
│  │  • Scoring Results & History                     │  │
│  │  • Rate Limits (Qwen 3.5)                       │  │
│  │  • Encrypted Sensitive Data                      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core
- **Backend API**: FastAPI (async, security, auto-docs)
- **Frontend**: Streamlit (maintains existing UI)
- **Database**: PostgreSQL or MySQL + SQLAlchemy ORM
- **Authentication**: JWT + bcrypt + email verification

### ML & Scoring
- **Old**: BQS (Google BigQuery Service)
- **New**: EQS with Qwen 3.5 (1k requests/period, rate-limited)
- **Fallback**: Keep existing transformer models as backup

### Security
- **Password**: bcrypt (12 rounds)
- **Data**: AES-256 for sensitive fields
- **API**: JWT tokens (15min access, 7day refresh)
- **Email**: SendGrid/Mailgun for verification

## Database Schema

### Core Tables
```
Users
├── id (UUID, PK)
├── username (unique, indexed)
├── email (unique, indexed)
├── password_hash (bcrypt)
├── email_verified (bool)
├── verification_token (encrypted)
├── profile_bio (string)
├── avatar_url (string)
├── created_at
└── updated_at

WriterProfiles
├── user_id (FK → Users)
├── writing_style (enum: analytical, balanced, emotional, etc.)
├── primary_topics (array: cricket_format, batting, bowling, etc.)
├── total_submissions (int)
├── avg_score (float)
├── reputation_points (int)
└── metadata (JSON)

Communities
├── id (UUID, PK)
├── name (string, unique)
├── slug (string, unique)
├── description (text)
├── primary_topic (string, indexed)
├── creator_id (FK → Users)
├── member_count (int)
├── rules (JSON)
├── visibility (enum: public, private)
├── created_at
└── updated_at

CommunityMembers
├── community_id (FK → Communities)
├── user_id (FK → Users)
├── role (enum: owner, moderator, member)
├── joined_at
└── last_active_at

ScoringResults
├── id (UUID, PK)
├── user_id (FK → Users)
├── community_id (FK → Communities, nullable)
├── text (text)
├── model_version (string)
├── bqs_score (float, legacy)
├── eqs_score (float)
├── confidence (float)
├── components (JSON)
├── created_at
└── request_timestamp (for rate limiting)

RateLimitLog
├── user_id (FK → Users)
├── model (string: 'qwen3.5')
├── request_count (int)
├── period_start (datetime)
├── period_end (datetime)
└── status (enum: active, warning, blocked)
```

## Features Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Database setup & migrations
- [ ] User authentication (signup/login/email verification)
- [ ] JWT token management
- [ ] FastAPI wrapper around existing Streamlit app

### Phase 2: User Management (Week 2-3)
- [ ] Writer profile creation
- [ ] User dashboard with history
- [ ] Password reset & account settings

### Phase 3: Communities (Week 3-4)
- [ ] Community CRUD operations
- [ ] Topic-based discovery
- [ ] Member management & roles

### Phase 4: EQS Migration (Week 4-5)
- [ ] Qwen 3.5 integration
- [ ] Rate limiting logic (1k requests)
- [ ] Model fallback system
- [ ] A/B testing: BQS vs EQS

### Phase 5: Security Hardening (Week 5-6)
- [ ] Data encryption at rest
- [ ] Input validation & sanitization
- [ ] API rate limiting
- [ ] Audit logging

## Security Considerations

1. **Authentication**
   - Passwords: bcrypt (12+ rounds)
   - Tokens: JWT with RS256 signing
   - Refresh: 7-day refresh tokens, 15-min access tokens

2. **Data Protection**
   - Sensitive fields encrypted with AES-256
   - HTTPS only (prod)
   - CORS restricted to trusted origins

3. **Database**
   - Connection pooling with timeout
   - Parameterized queries (SQLAlchemy)
   - Row-level security for user data
   - Regular backups

4. **Rate Limiting**
   - Per-user Qwen 3.5 quota (1k/period)
   - Per-IP API rate limit (100/min)
   - Exponential backoff for retries

## Environment Configuration

```env
# Database
DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/cricgeek?charset=utf8mb4

# JWT
JWT_SECRET_KEY=<64-char random string>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Qwen 3.5
QWEN_API_KEY=<api key>
QWEN_RATE_LIMIT_REQUESTS=1000
QWEN_RATE_LIMIT_PERIOD_DAYS=30

# Email
EMAIL_PROVIDER=sendgrid  # or mailgun
EMAIL_API_KEY=<api key>

# Security
ENCRYPTION_KEY=<32-char key for AES-256>
BCRYPT_ROUNDS=12
```

## Testing Strategy

- Unit tests for auth, database, business logic
- Integration tests for API endpoints
- Load tests for rate limiting (1k requests)
- Security tests (SQL injection, XSS, CSRF)

## Deployment

- Docker containerization
- PostgreSQL on managed service (AWS RDS / Heroku)
- FastAPI on cloud (Heroku / Railway / Azure)
- Streamlit frontend on separate container
- CI/CD with GitHub Actions

## Migration Path from BQS to EQS

1. **Phase 1**: Run both in parallel (hybrid mode)
   - Store both BQS & EQS scores
   - Compare quality metrics
   - Build confidence in EQS

2. **Phase 2**: Gradual rollout
   - Route 10% → EQS, 90% → BQS
   - Monitor error rates & performance
   - Increment % daily

3. **Phase 3**: Full cutover
   - All requests to EQS
   - Keep BQS as emergency fallback
   - Archive historical data

4. **Phase 4**: Cleanup
   - Remove BigQuery integrations
   - Optimize Qwen 3.5 costs
   - Archive BQS results
