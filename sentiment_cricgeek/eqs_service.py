"""
Qwen 3.5 integration for Enhanced Query Service (EQS)
Temporary Qwen-backed Expression Quality Score with rate limiting and fallback support.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

import requests
from sqlalchemy.orm import Session

from models import ScoringResult, RateLimitLog, RateLimitStatusEnum


# Qwen API Configuration
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-3.5")
QWEN_RATE_LIMIT_REQUESTS = int(os.getenv("QWEN_RATE_LIMIT_REQUESTS", "1000"))
QWEN_RATE_LIMIT_PERIOD_DAYS = int(os.getenv("QWEN_RATE_LIMIT_PERIOD_DAYS", "30"))
QWEN_TIMEOUT_SECONDS = int(os.getenv("QWEN_TIMEOUT_SECONDS", "30"))


class EQSService:
    """
    Expression Quality Score service backed by Qwen while the local pretrained
    model is being trained.

    It enforces a per-user Qwen quota, falls back to the local deterministic
    scoring pipeline, and returns a stable EQS response shape.
    """
    
    @staticmethod
    def check_rate_limit(user_id: str, db: Session) -> Tuple[bool, str]:
        """
        Check if user has exceeded rate limit for Qwen 3.5
        
        Args:
            user_id: User UUID
            db: Database session
            
        Returns:
            Tuple of (is_allowed, message)
        """
        # Get or create rate limit log for current period
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=QWEN_RATE_LIMIT_PERIOD_DAYS)
        
        rate_limit = db.query(RateLimitLog).filter(
            RateLimitLog.user_id == user_id,
            RateLimitLog.model == QWEN_MODEL,
            RateLimitLog.period_start == period_start,
        ).first()
        
        if not rate_limit:
            # Create new rate limit log for this period
            rate_limit = RateLimitLog(
                user_id=user_id,
                model=QWEN_MODEL,
                request_count=0,
                max_requests=QWEN_RATE_LIMIT_REQUESTS,
                period_start=period_start,
                period_end=period_end,
                status=RateLimitStatusEnum.ACTIVE,
            )
            db.add(rate_limit)
            db.commit()
        
        # Check limit
        if rate_limit.is_over_limit():
            rate_limit.status = RateLimitStatusEnum.BLOCKED
            db.commit()
            return False, f"Rate limit exceeded: {rate_limit.max_requests} requests per {QWEN_RATE_LIMIT_PERIOD_DAYS} days"
        
        # Warn if approaching limit
        if rate_limit.should_warn():
            usage = rate_limit.get_usage_percentage()
            remaining = rate_limit.max_requests - rate_limit.request_count
            rate_limit.status = RateLimitStatusEnum.WARNING
            db.commit()
            return True, f"Warning: {remaining} requests remaining ({usage:.0f}% used)"
        
        # OK to proceed
        rate_limit.status = RateLimitStatusEnum.ACTIVE
        db.commit()
        return True, "OK"
    
    @staticmethod
    def increment_request_count(user_id: str, db: Session) -> None:
        """Increment rate limit counter"""
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        rate_limit = db.query(RateLimitLog).filter(
            RateLimitLog.user_id == user_id,
            RateLimitLog.model == QWEN_MODEL,
            RateLimitLog.period_start == period_start,
        ).first()
        
        if rate_limit:
            rate_limit.request_count += 1
            db.commit()
    
    @staticmethod
    def score_text(
        text: str,
        user_id: str,
        db: Session,
        community_id: Optional[str] = None,
        use_legacy_bqs: bool = False,
    ) -> Dict[str, Any]:
        """
        Score text using Qwen 3.5 (or fallback to the local EQS pipeline)
        
        Args:
            text: Text to score
            user_id: User ID for rate limiting
            db: Database session
            community_id: Optional community context
            use_legacy_bqs: Force use of the local scoring pipeline
            
        Returns:
            Scoring result dictionary
        """
        start_time = time.time()
        
        # Check rate limit
        allowed, limit_msg = EQSService.check_rate_limit(user_id, db)
        if not allowed and not use_legacy_bqs:
            return {
                "success": False,
                "error": limit_msg,
                "eqs_score": None,
                "fallback_to_bqs": False,
            }
        
        try:
            if use_legacy_bqs:
                # Use local scoring pipeline
                result = EQSService._score_with_local_pipeline(text)
                response_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "success": True,
                    "model_version": "EQS-local",
                    "bqs_score": None,
                    "eqs_score": result.get("score"),  # Map to EQS field
                    "confidence": result.get("confidence", 0.5),
                    "components": result.get("components", {}),
                    "response_time_ms": response_time_ms,
                    "fallback_to_bqs": False,
                }
            
            else:
                # Use Qwen 3.5 EQS
                result = EQSService._score_with_qwen3_5(text)
                response_time_ms = int((time.time() - start_time) * 1000)
                
                if result.get("success"):
                    # Increment counter on success
                    EQSService.increment_request_count(user_id, db)
                    
                    return {
                        "success": True,
                        "model_version": f"EQS-{QWEN_MODEL}-v1",
                        "bqs_score": None,
                        "eqs_score": result.get("score"),
                        "confidence": result.get("confidence"),
                        "components": result.get("components", {}),
                        "response_time_ms": response_time_ms,
                        "fallback_to_bqs": False,
                    }
                else:
                    # Fallback to local scorer on failure
                    print(f"[EQS] Qwen failed, falling back to local EQS: {result.get('error')}")
                    result = EQSService._score_with_local_pipeline(text)
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    return {
                        "success": True,
                        "model_version": "EQS-local-fallback",
                        "bqs_score": None,
                        "eqs_score": result.get("score"),  # Map to EQS field
                        "confidence": result.get("confidence", 0.5),
                        "components": result.get("components", {}),
                        "response_time_ms": response_time_ms,
                        "fallback_to_bqs": True,
                    }
        
        except Exception as e:
            print(f"[EQS] Error during scoring: {e}")
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Last resort fallback to local EQS
            try:
                result = EQSService._score_with_local_pipeline(text)
                return {
                    "success": True,
                    "model_version": "EQS-local-fallback",
                    "bqs_score": None,
                    "eqs_score": result.get("score"),
                    "confidence": result.get("confidence", 0.5),
                    "components": result.get("components", {}),
                    "response_time_ms": response_time_ms,
                    "fallback_to_bqs": True,
                }
            except Exception as e2:
                return {
                    "success": False,
                    "error": f"All scoring methods failed: {str(e2)}",
                    "eqs_score": None,
                    "fallback_to_bqs": False,
                }
    
    @staticmethod
    def _score_with_qwen3_5(text: str) -> Dict[str, Any]:
        """
        Score using the configured Qwen-compatible chat completions API.
        """
        if not QWEN_API_KEY:
            return {"success": False, "error": "QWEN_API_KEY is not configured"}

        system_prompt = (
            "You are CricGeek's Expression Quality Score evaluator for cricket writing. "
            "Score the writing for clarity, argument quality, evidence, constructiveness, "
            "toxicity, and cricket-specific reasoning. Return only valid JSON."
        )
        user_prompt = {
            "task": "Score this cricket article/commentary from 20 to 95.",
            "text": text,
            "required_json_schema": {
                "score": "float from 20 to 95",
                "confidence": "float from 0 to 1",
                "components": {
                    "stance_score": "float from 0 to 100",
                    "stance_label": "balanced|positive|negative|analytical|dismissive|attack_based",
                    "stance_confidence": "float from 0 to 1",
                    "stats_verified": "boolean",
                    "writing_quality_score": "float from 0 to 100",
                    "toxicity_score": "float from 0 to 100",
                    "constructiveness_score": "float from 0 to 100"
                }
            },
        }

        try:
            response = requests.post(
                QWEN_API_URL,
                headers={
                    "Authorization": f"Bearer {QWEN_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": QWEN_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_prompt)},
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=QWEN_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            score = EQSService._clamp_float(parsed.get("score"), 20.0, 95.0, 50.0)
            confidence = EQSService._clamp_float(parsed.get("confidence"), 0.0, 1.0, 0.7)
            raw_components = parsed.get("components", {})
            components = raw_components if isinstance(raw_components, dict) else {}

            return {
                "success": True,
                "score": score,
                "confidence": confidence,
                "components": {
                    "stance_score": EQSService._clamp_float(components.get("stance_score"), 0.0, 100.0, 50.0),
                    "stance_label": str(components.get("stance_label", "balanced")),
                    "stance_confidence": EQSService._clamp_float(components.get("stance_confidence"), 0.0, 1.0, confidence),
                    "stats_verified": bool(components.get("stats_verified", False)),
                    "writing_quality_score": EQSService._clamp_float(components.get("writing_quality_score"), 0.0, 100.0, score),
                    "toxicity_score": EQSService._clamp_float(components.get("toxicity_score"), 0.0, 100.0, 0.0),
                    "constructiveness_score": EQSService._clamp_float(components.get("constructiveness_score"), 0.0, 100.0, 50.0),
                },
            }
        
        except Exception as e:
            print(f"[Qwen3.5] API Error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, number))
    
    @staticmethod
    def _score_with_local_pipeline(text: str) -> Dict[str, Any]:
        """
        Score using the local deterministic sentiment pipeline.
        
        This integrates with the existing sentiment_engine
        """
        try:
            from sentiment_engine.sentiment_pipeline import SentimentPipeline
            
            pipeline = SentimentPipeline()
            result = pipeline.score(text)
            
            return {
                "success": True,
                "score": result.get("final_score", 50.0),
                "confidence": result.get("confidence", 0.7),
                "components": result,
            }
        
        except ImportError:
            # Sentiment pipeline not available
            print("[EQS-local] Sentiment pipeline not available")
            return {
                "success": False,
                "error": "Sentiment pipeline not available",
                "components": {
                    "stance_label": "unknown",
                }
            }
        except Exception as e:
            print(f"[EQS-local] Scoring Error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def get_rate_limit_status(user_id: str, db: Session) -> Dict[str, Any]:
        """Get current rate limit status for user"""
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        rate_limit = db.query(RateLimitLog).filter(
            RateLimitLog.user_id == user_id,
            RateLimitLog.model == QWEN_MODEL,
            RateLimitLog.period_start == period_start,
        ).first()
        
        if not rate_limit:
            return {
                "model": QWEN_MODEL,
                "request_count": 0,
                "max_requests": QWEN_RATE_LIMIT_REQUESTS,
                "usage_percentage": 0.0,
                "status": "active",
                "requests_remaining": QWEN_RATE_LIMIT_REQUESTS,
            }
        
        return {
            "model": rate_limit.model,
            "request_count": rate_limit.request_count,
            "max_requests": rate_limit.max_requests,
            "usage_percentage": rate_limit.get_usage_percentage(),
            "status": rate_limit.status.value,
            "requests_remaining": rate_limit.max_requests - rate_limit.request_count,
            "period_start": rate_limit.period_start.isoformat(),
            "period_end": rate_limit.period_end.isoformat(),
        }
