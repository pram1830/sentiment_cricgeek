# CricGeek Platform v2.0 - Setup Guide

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ or MySQL 8+
- Docker (optional)

PostgreSQL and MySQL are supported database choices for CricGeek. Use either one for production or local development.
MySQL is the default local setup in this repository, and PostgreSQL remains supported if you prefer it.

### Installation

#### 1. **Clone Repository**
```bash
git clone <repo-url>
cd sentiment_cricgeek
```

#### 2. **Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Configure Environment**

Create `.env` file in project root:

```env
# Database
DATABASE_URL=mysql+pymysql://cricgeek:password@localhost:3306/cricgeek_dev?charset=utf8mb4

# PostgreSQL alternative
# DATABASE_URL=postgresql://cricgeek:password@localhost:5432/cricgeek_dev

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-key-generate-with-secrets.token_urlsafe(32)
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Qwen 3.5 API
QWEN_API_KEY=your-qwen-api-key
QWEN_RATE_LIMIT_REQUESTS=1000
QWEN_RATE_LIMIT_PERIOD_DAYS=30

# Email Configuration
EMAIL_PROVIDER=console  # or sendgrid, mailgun
EMAIL_API_KEY=your-email-api-key
SENDER_EMAIL=noreply@cricgeek.local

# Security
ENCRYPTION_KEY=your-32-byte-base64-encoded-key
BCRYPT_ROUNDS=12

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501

# Environment
ENVIRONMENT=development
```

#### 5. **Setup Database**
```bash
# Initialize database schema
python database.py init

# (Optional) Seed with sample data
python database.py seed
```

#### 6. **Start FastAPI Backend**
```bash
# Development
uvicorn main_api:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main_api:app --bind 0.0.0.0:8000
```

#### 7. **Start Streamlit Frontend (in another terminal)**
```bash
streamlit run app.py --server.port 8501
```

Access the app at: **http://localhost:8501**

### Using MySQL with Docker

If you want a fully containerized MySQL setup, use the dedicated compose file:

```bash
docker compose -f docker-compose.mysql.yml up --build
```

That starts MySQL, runs `python database.py init`, seeds the schema, and then launches the API and Streamlit app.

---

## API Documentation

Once FastAPI backend is running, visit:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

---

## Architecture Overview

### Components

```
┌─────────────────────────┐
│   Streamlit Frontend    │
│  (Port 8501)            │
│  - Auth Pages           │
│  - Dashboard            │
│  - Communities          │
└────────────┬────────────┘
             │ HTTP Calls
             ↓
┌─────────────────────────┐
│    FastAPI Backend      │
│  (Port 8000)            │
│  - Auth Endpoints       │
│  - Scoring Endpoints    │
│  - Community Endpoints  │
└────────────┬────────────┘
             │ SQL
             ↓
┌─────────────────────────┐
│   PostgreSQL Database   │
│  (Port 5432)            │
└─────────────────────────┘
```

### Database Schema

**Core Tables:**
- `users` - User accounts & authentication
- `refresh_tokens` - JWT token tracking
- `writer_profiles` - Writer statistics & preferences
- `communities` - Community metadata
- `community_members` - Membership tracking
- `scoring_results` - Scoring history
- `rate_limit_logs` - Qwen 3.5 quota tracking

## Storage Security Baseline

- Keep `ENVIRONMENT=production` paired with a PostgreSQL `DATABASE_URL`; SQLite is development-only.
- Store passwords as bcrypt hashes and sensitive tokens as hashed lookup values.
- Keep `ENCRYPTION_KEY` in a secret manager or protected environment variable, not in source control.
- Use the PostgreSQL service from `docker-compose.yml` for local integration testing to match production behavior.
- Schedule database backups and restrict DB credentials to the minimum required privileges.

---

## Features

### 🔐 Authentication
- ✅ Signup with email verification
- ✅ Login/Logout
- ✅ Password reset
- ✅ JWT token management
- ✅ Email verification tokens

### 📊 EQS Scoring (Qwen 3.5)
- ✅ Text scoring via Qwen 3.5
- ✅ Rate limiting (1000 requests/30 days)
- ✅ Fallback to legacy BQS
- ✅ Scoring history
- ✅ Rate limit status dashboard

### 👥 Communities
- ✅ Create/Browse communities
- ✅ Topic-based discovery
- ✅ Member management
- ✅ Community recommendations
- ✅ Scoring context (community-specific results)

### 👤 User Management
- ✅ Profile customization
- ✅ Writer profile with topics
- ✅ Account settings
- ✅ Writer statistics tracking

---

## BQS to EQS Migration

### Phase 1: Parallel Running (Current)
Both BQS and EQS score all submissions, allowing quality comparison.

### Phase 2: Gradual Rollout (Next)
```
Day 1-2:  10% → EQS, 90% → BQS
Day 3-4:  25% → EQS, 75% → BQS
Day 5-6:  50% → EQS, 50% → BQS
Day 7-8:  75% → EQS, 25% → BQS
Day 9-10: 100% → EQS, BQS as fallback
```

### Phase 3: Full Cutover
All new submissions use EQS, BQS available as emergency fallback.

### Configuration
```python
# In eqs_service.py, use_legacy_bqs parameter
EQSService.score_text(
    text=text,
    user_id=user_id,
    db=db,
    use_legacy_bqs=False  # Set to True to force BQS
)
```

---

## Database Migrations

Using Alembic for schema versioning:

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Revert to previous version
alembic downgrade -1
```

---

## Security Checklist

- [ ] **Database**
  - [ ] Change default passwords
  - [ ] Enable SSL connections
  - [ ] Regular backups configured
  - [ ] Row-level security policies

- [ ] **API**
  - [ ] HTTPS enabled (production)
  - [ ] CORS properly configured
  - [ ] Rate limiting active (per-IP + per-user)
  - [ ] API keys secured in environment

- [ ] **Authentication**
  - [ ] JWT secret is strong (32+ chars)
  - [ ] Token expiration reasonable (15 min access)
  - [ ] Refresh token rotation enabled
  - [ ] Password hashing with bcrypt (12 rounds)

- [ ] **Encryption**
  - [ ] AES-256 for sensitive fields
  - [ ] Encryption key secured (environment/vault)
  - [ ] Encrypted database connection

- [ ] **Email**
  - [ ] Email provider configured (SendGrid/Mailgun)
  - [ ] Email templates reviewed
  - [ ] Bounce handling configured

---

## Rate Limiting Details

### Qwen 3.5 Limits
- **Quota**: 1000 requests per 30 days (configurable)
- **Status Levels**:
  - `active`: < 80% usage
  - `warning`: 80-99% usage
  - `blocked`: 100%+ usage

### API Limits (Per IP)
- **General**: 100 requests/minute
- **Auth**: 5 login attempts/5 minutes
- **Scoring**: 10 requests/minute

---

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
psql -U cricgeek -d cricgeek_dev -c "SELECT 1"

# View SQLAlchemy logs
# Set echo=True in database.py engine config
```

### JWT Token Issues
```bash
# Verify token format
python -c "import jwt; print(jwt.decode('<token>', 'secret', algorithms=['HS256']))"
```

### API Not Responding
```bash
# Check if FastAPI is running
curl http://localhost:8000/health

# View logs
tail -f uvicorn.log
```

### Email Not Sending
- Check `EMAIL_PROVIDER` in `.env`
- Verify API key is correct
- For console: check terminal output

---

## Performance Optimization

### Database
- Add indexes on frequently queried columns (done in models)
- Use connection pooling (configured in database.py)
- Regular VACUUM/ANALYZE for PostgreSQL

### API
- Enable gzip compression
- Implement request caching for public endpoints
- Use async endpoints where possible

### Streamlit
- Use `@st.cache_resource` for heavy objects
- Optimize pandas operations
- Lazy load images/data

---

## Deployment Options

### 1. **Heroku** (PaaS)
```bash
heroku create cricgeek-prod
heroku addons:create heroku-postgresql:standard-0
git push heroku main
```

### 2. **Railway** (PaaS)
```bash
railway up
```

### 3. **AWS** (IaaS)
- Streamlit on EC2 / ECS
- FastAPI on ECS / Lambda
- PostgreSQL on RDS
- Route53 for DNS

### 4. **Docker** (Self-hosted)
```bash
docker-compose up
```

See `docker-compose.yml` for configuration.

---

## Monitoring

### Metrics to Track
- Request latency (p50, p95, p99)
- Error rates by endpoint
- Rate limit hit frequency
- Scoring model accuracy (EQS vs BQS)
- Database connection pool usage
- Qwen 3.5 API quota usage

### Logging
- Application logs: `logs/app.log`
- API requests: `logs/api.log`
- Database queries: `logs/database.log`

### Alerts
- Error rate > 5%
- API response time > 5s
- Database connections > 80%
- Rate limit quota > 90%

---

## Support

For issues or questions:
1. Check logs for error messages
2. Review API docs at `/api/docs`
3. Check database integrity: `SELECT * FROM pg_stat_user_tables`
4. Contact: support@cricgeek.local

---

## License

See LICENSE file for details.

---

## Version History

### v2.0.0 (Current)
- ✅ FastAPI backend architecture
- ✅ PostgreSQL database with ORM
- ✅ JWT authentication system
- ✅ Community features
- ✅ Qwen 3.5 integration with rate limiting
- ✅ Streamlit frontend with auth pages

### v1.0.0 (Legacy)
- BQS (BigQuery Service) based scoring
- Streamlit dashboard only
- No user authentication
