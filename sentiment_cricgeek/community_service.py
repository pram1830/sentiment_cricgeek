"""
Community management service - handle community CRUD, membership, discovery
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from models import (
    Community, CommunityMember, User, TopicEnum, CommunityVisibilityEnum, 
    MemberRoleEnum, WriterProfile
)


class CommunityService:
    """Manage community operations"""

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)
    
    @staticmethod
    def create_community(
        name: str,
        slug: str,
        description: Optional[str],
        primary_topic: str,
        secondary_topics: List[str],
        creator_id: UUID,
        visibility: str,
        db: Session,
    ) -> Tuple[bool, str, Optional[Community]]:
        """
        Create a new community
        
        Args:
            name: Community name
            slug: URL-friendly slug
            description: Community description
            primary_topic: Primary topic (TopicEnum)
            secondary_topics: List of secondary topics
            creator_id: User ID of creator
            visibility: public, private, or invite_only
            db: Database session
            
        Returns:
            Tuple of (success, message, community_object)
        """
        # Check if slug already exists
        if db.query(Community).filter(Community.slug == slug).first():
            return False, "Slug already exists", None
        
        try:
            community = Community(
                name=name,
                slug=slug,
                description=description,
                primary_topic=primary_topic,
                secondary_topics=secondary_topics,
                creator_id=creator_id,
                visibility=visibility,
                member_count=1,  # Creator is first member
            )
            
            db.add(community)
            db.flush()
            
            # Add creator as owner
            member = CommunityMember(
                community_id=community.id,
                user_id=creator_id,
                role=MemberRoleEnum.OWNER,
            )
            db.add(member)

            CommunityService._ensure_creator_topics(
                creator_id=creator_id,
                topics=[primary_topic] + list(secondary_topics or []),
                db=db,
            )

            db.commit()
            db.refresh(community)

            return True, f"Community '{name}' created successfully", community
        
        except Exception as e:
            db.rollback()
            return False, f"Failed to create community: {str(e)}", None

    @staticmethod
    def _ensure_creator_topics(creator_id: UUID, topics: List[str], db: Session) -> None:
        """Keep a creator's writer profile connected to topics they build around."""
        normalized_topics = [
            CommunityService._enum_value(topic)
            for topic in topics
            if CommunityService._enum_value(topic)
        ]
        if not normalized_topics:
            return

        writer_profile = db.query(WriterProfile).filter(
            WriterProfile.user_id == creator_id
        ).first()

        if not writer_profile:
            writer_profile = WriterProfile(
                user_id=creator_id,
                primary_topics=[],
            )
            db.add(writer_profile)
            db.flush()

        current_topics = list(writer_profile.primary_topics or [])
        for topic in normalized_topics:
            if topic not in current_topics:
                current_topics.append(topic)

        writer_profile.primary_topics = current_topics[:5]
        writer_profile.updated_at = datetime.utcnow()
    
    @staticmethod
    def get_community(community_id: UUID, db: Session) -> Optional[Community]:
        """Get community by ID"""
        return db.query(Community).filter(Community.id == community_id).first()
    
    @staticmethod
    def get_community_by_slug(slug: str, db: Session) -> Optional[Community]:
        """Get community by slug"""
        return db.query(Community).filter(Community.slug == slug).first()
    
    @staticmethod
    def list_communities(
        topic: Optional[str] = None,
        visibility: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        db: Session = None,
    ) -> Tuple[int, List[Community]]:
        """
        List communities with filters
        
        Returns:
            Tuple of (total_count, communities_list)
        """
        query = db.query(Community)
        
        # Filter by topic
        if topic:
            query = query.filter(
                or_(
                    Community.primary_topic == topic,
                    Community.secondary_topics.contains([topic])
                )
            )
        
        # Filter by visibility
        if visibility:
            query = query.filter(Community.visibility == visibility)
        
        # Search by name or description
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Community.name.ilike(search_term),
                    Community.description.ilike(search_term),
                )
            )
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination
        communities = query.order_by(Community.created_at.desc()).offset(offset).limit(limit).all()
        
        return total_count, communities
    
    @staticmethod
    def update_community(
        community_id: UUID,
        user_id: UUID,
        description: Optional[str] = None,
        visibility: Optional[str] = None,
        rules: Optional[Dict[str, Any]] = None,
        db: Session = None,
    ) -> Tuple[bool, str]:
        """
        Update community (creator/owner only)
        
        Returns:
            Tuple of (success, message)
        """
        community = db.query(Community).filter(Community.id == community_id).first()
        
        if not community:
            return False, "Community not found"
        
        # Check if user is owner
        if community.creator_id != user_id:
            member = db.query(CommunityMember).filter(
                and_(
                    CommunityMember.community_id == community_id,
                    CommunityMember.user_id == user_id,
                    CommunityMember.role == MemberRoleEnum.OWNER,
                )
            ).first()
            if not member:
                return False, "Only community owners can edit settings"
        
        # Update fields
        if description is not None:
            community.description = description
        if visibility is not None:
            community.visibility = visibility
        if rules is not None:
            community.rules = rules
        
        community.updated_at = datetime.utcnow()
        db.commit()
        
        return True, "Community updated"
    
    @staticmethod
    def delete_community(
        community_id: UUID,
        user_id: UUID,
        db: Session,
    ) -> Tuple[bool, str]:
        """
        Delete community (owner only)
        
        Returns:
            Tuple of (success, message)
        """
        community = db.query(Community).filter(Community.id == community_id).first()
        
        if not community:
            return False, "Community not found"
        
        if community.creator_id != user_id:
            return False, "Only the creator can delete this community"
        
        try:
            db.delete(community)
            db.commit()
            return True, "Community deleted"
        except Exception as e:
            db.rollback()
            return False, f"Failed to delete community: {str(e)}"
    
    @staticmethod
    def join_community(
        community_id: UUID,
        user_id: UUID,
        db: Session,
    ) -> Tuple[bool, str]:
        """
        Join a community
        
        Returns:
            Tuple of (success, message)
        """
        community = db.query(Community).filter(Community.id == community_id).first()
        
        if not community:
            return False, "Community not found"
        
        # Check if already member
        existing = db.query(CommunityMember).filter(
            and_(
                CommunityMember.community_id == community_id,
                CommunityMember.user_id == user_id,
            )
        ).first()
        
        if existing:
            return False, "Already a member of this community"
        
        # Check privacy
        visibility = CommunityService._enum_value(community.visibility)
        if visibility in (
            CommunityVisibilityEnum.PRIVATE.value,
            CommunityVisibilityEnum.INVITE_ONLY.value,
        ):
            return False, "This community is private. Request to join."
        
        try:
            member = CommunityMember(
                community_id=community_id,
                user_id=user_id,
                role=MemberRoleEnum.MEMBER,
            )
            db.add(member)
            community.member_count += 1
            db.commit()
            return True, "Successfully joined community"
        except Exception as e:
            db.rollback()
            return False, f"Failed to join community: {str(e)}"
    
    @staticmethod
    def leave_community(
        community_id: UUID,
        user_id: UUID,
        db: Session,
    ) -> Tuple[bool, str]:
        """
        Leave a community
        
        Returns:
            Tuple of (success, message)
        """
        member = db.query(CommunityMember).filter(
            and_(
                CommunityMember.community_id == community_id,
                CommunityMember.user_id == user_id,
            )
        ).first()
        
        if not member:
            return False, "Not a member of this community"
        
        # Owner cannot leave
        if member.role == MemberRoleEnum.OWNER:
            return False, "Owner cannot leave. Transfer ownership first."
        
        try:
            db.delete(member)
            community = db.query(Community).filter(Community.id == community_id).first()
            if community:
                community.member_count = max(0, community.member_count - 1)
            db.commit()
            return True, "Successfully left community"
        except Exception as e:
            db.rollback()
            return False, f"Failed to leave community: {str(e)}"
    
    @staticmethod
    def get_community_members(
        community_id: UUID,
        limit: int = 100,
        offset: int = 0,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """Get community members"""
        members = db.query(CommunityMember).filter(
            CommunityMember.community_id == community_id
        ).offset(offset).limit(limit).all()
        
        result = []
        for member in members:
            user = db.query(User).filter(User.id == member.user_id).first()
            if user:
                result.append({
                    "user_id": str(user.id),
                    "username": user.username,
                    "role": member.role.value,
                    "joined_at": member.joined_at.isoformat(),
                    "avatar_url": user.avatar_url,
                })
        
        return result
    
    @staticmethod
    def discover_communities_for_writer(
        user_id: UUID,
        limit: int = 10,
        db: Session = None,
    ) -> List[Community]:
        """
        Discover communities matching writer's interests
        """
        # Get writer's topics
        writer_profile = db.query(WriterProfile).filter(
            WriterProfile.user_id == user_id
        ).first()
        
        if not writer_profile or not writer_profile.primary_topics:
            # No preferences, return recent public communities
            return db.query(Community).filter(
                Community.visibility == CommunityVisibilityEnum.PUBLIC
            ).order_by(Community.created_at.desc()).limit(limit).all()
        
        # Find communities matching topics
        topics = writer_profile.primary_topics[:3]  # Top 3 topics
        communities = db.query(Community).filter(
            and_(
                Community.visibility == CommunityVisibilityEnum.PUBLIC,
                or_(
                    Community.primary_topic.in_(topics),
                    *[Community.secondary_topics.contains([t]) for t in topics]
                )
            )
        ).order_by(Community.member_count.desc()).limit(limit).all()
        
        return communities

    @staticmethod
    def find_like_minded_writers(
        user_id: UUID,
        topic: Optional[str] = None,
        writing_style: Optional[str] = None,
        limit: int = 20,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """Find writers with overlapping topics and, when available, style."""
        topic_value = CommunityService._enum_value(topic)
        style_value = CommunityService._enum_value(writing_style)

        current_profile = db.query(WriterProfile).filter(
            WriterProfile.user_id == user_id
        ).first()
        preferred_topics = []
        if current_profile and current_profile.primary_topics:
            preferred_topics = list(current_profile.primary_topics)

        target_topics = [topic_value] if topic_value else preferred_topics

        profiles = db.query(WriterProfile).join(User).filter(
            WriterProfile.user_id != user_id,
            User.is_active == True,
            User.email_verified == True,
        ).all()

        matches: List[Dict[str, Any]] = []
        for profile in profiles:
            profile_topics = list(profile.primary_topics or [])
            shared_topics = [t for t in target_topics if t in profile_topics] if target_topics else profile_topics

            profile_style = CommunityService._enum_value(profile.writing_style)
            style_matches = bool(style_value and profile_style == style_value)
            if not shared_topics and not style_matches:
                continue

            match_score = len(shared_topics) * 2 + (1 if style_matches else 0)
            matches.append({
                "user_id": str(profile.user.id),
                "username": profile.user.username,
                "avatar_url": profile.user.avatar_url,
                "writing_style": profile_style,
                "primary_topics": profile_topics,
                "shared_topics": shared_topics,
                "avg_eqs_score": profile.avg_eqs_score,
                "total_submissions": profile.total_submissions,
                "reputation_points": profile.reputation_points,
                "match_score": match_score,
                "match_reason": CommunityService._build_writer_match_reason(shared_topics, style_matches),
            })

        matches.sort(
            key=lambda item: (
                item["match_score"],
                item.get("reputation_points") or 0,
                item.get("total_submissions") or 0,
            ),
            reverse=True,
        )
        return matches[:limit]

    @staticmethod
    def _build_writer_match_reason(shared_topics: List[str], style_matches: bool) -> str:
        reasons = []
        if shared_topics:
            reasons.append("same topic: " + ", ".join(shared_topics[:3]))
        if style_matches:
            reasons.append("similar writing style")
        return "; ".join(reasons) if reasons else "similar CricGeek interests"
    
    @staticmethod
    def get_user_communities(
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
        db: Session = None,
    ) -> List[Community]:
        """Get communities user is member of"""
        communities = db.query(Community).join(CommunityMember).filter(
            CommunityMember.user_id == user_id
        ).offset(offset).limit(limit).all()
        
        return communities
