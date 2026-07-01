"""
Durable storage helpers for writer submissions and profile statistics.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from models import Community, ScoringResult, WriterProfile, WritingStyleEnum


class WriterDataService:
    """Persist scored writing and keep writer profile aggregates current."""

    @staticmethod
    def record_scoring_result(
        *,
        user_id: UUID,
        text: str,
        scoring_result: Dict[str, Any],
        db: Session,
        community_id: Optional[str] = None,
    ) -> ScoringResult:
        normalized_community_id = WriterDataService._normalize_uuid(community_id)
        result_obj = ScoringResult(
            user_id=user_id,
            community_id=normalized_community_id,
            text=text,
            text_length=len(text),
            model_version=scoring_result.get("model_version"),
            bqs_score=scoring_result.get("bqs_score"),
            eqs_score=scoring_result.get("eqs_score"),
            confidence=scoring_result.get("confidence"),
            components=scoring_result.get("components", {}),
            response_time_ms=scoring_result.get("response_time_ms"),
        )
        db.add(result_obj)

        WriterDataService._update_writer_profile(
            user_id=user_id,
            scoring_result=scoring_result,
            db=db,
            community_id=normalized_community_id,
        )

        db.commit()
        db.refresh(result_obj)
        return result_obj

    @staticmethod
    def _update_writer_profile(
        *,
        user_id: UUID,
        scoring_result: Dict[str, Any],
        db: Session,
        community_id: Optional[UUID],
    ) -> None:
        profile = db.query(WriterProfile).filter(WriterProfile.user_id == user_id).first()
        if not profile:
            profile = WriterProfile(user_id=user_id, primary_topics=[], profile_metadata={})
            db.add(profile)
            db.flush()

        previous_total = profile.total_submissions or 0
        new_total = previous_total + 1

        eqs_score = scoring_result.get("eqs_score")
        bqs_score = scoring_result.get("bqs_score")
        profile.avg_eqs_score = WriterDataService._rolling_average(
            profile.avg_eqs_score,
            eqs_score,
            previous_total,
        )
        profile.avg_bqs_score = WriterDataService._rolling_average(
            profile.avg_bqs_score,
            bqs_score,
            previous_total,
        )
        profile.total_submissions = new_total

        components = scoring_result.get("components") or {}
        stance_label = components.get("stance_label")
        if stance_label in {style.value for style in WritingStyleEnum}:
            profile.writing_style = stance_label

        topics = list(profile.primary_topics or [])
        community = None
        if community_id:
            community = db.query(Community).filter(Community.id == community_id).first()
        if community:
            for topic in [community.primary_topic] + list(community.secondary_topics or []):
                topic_value = getattr(topic, "value", topic)
                if topic_value and topic_value not in topics:
                    topics.append(topic_value)
        profile.primary_topics = topics[:5]

        profile.reputation_points = max(
            profile.reputation_points or 0,
            int((profile.avg_eqs_score or 0) * new_total / 10),
        )
        profile.profile_metadata = {
            **(profile.profile_metadata or {}),
            "last_scored_at": datetime.utcnow().isoformat(),
            "last_model_version": scoring_result.get("model_version"),
            "last_confidence": scoring_result.get("confidence"),
        }
        profile.updated_at = datetime.utcnow()

    @staticmethod
    def _rolling_average(
        previous_average: Optional[float],
        new_value: Optional[float],
        previous_total: int,
    ) -> Optional[float]:
        if new_value is None:
            return previous_average
        if previous_average is None or previous_total == 0:
            return float(new_value)
        return ((previous_average * previous_total) + float(new_value)) / (previous_total + 1)

    @staticmethod
    def _normalize_uuid(value: Optional[str]) -> Optional[UUID]:
        if value is None or isinstance(value, UUID):
            return value
        return UUID(str(value))
