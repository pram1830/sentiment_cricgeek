from __future__ import annotations

from typing import Callable

from .constructiveness_detector import ConstructivenessDetector
from .bqs_aggregator import aggregate_bqs
from .model_loader import ModelLoader
from .paragraph_splitter import split_into_paragraphs
from .scoring_rules import analyze_paragraphs, apply_adaptive_scoring, apply_stance_aware_weighting, apply_toxicity_penalty
from .stats_verifier_csv import verify_blog_statistics
from .stance_score_calibration import apply_stance_score_calibration
from .stance_detector import StanceDetector
from .writer_dna_classifier import WriterDNAClassifier
from .writing_quality_layer import compute_writing_quality_component


class SentimentPipeline:
    """End-to-end writer-aware calibrated sentiment pipeline."""

    def __init__(self) -> None:
        self.model_loader = ModelLoader()
        self.writer_dna_classifier = WriterDNAClassifier()
        self.constructiveness_detector = ConstructivenessDetector()
        self.stance_detector = StanceDetector()

    def score(self, blog_text: str, enable_logs: bool = False, logger: Callable[[str], None] = print):
        if enable_logs:
            logger("Setup: loading models")
        models = self.model_loader.load()

        if enable_logs:
            logger("Setup: splitting paragraphs")
        paragraphs = split_into_paragraphs(blog_text)

        if enable_logs:
            logger("Step 1/10: Detect stance")
        stance_result = self.stance_detector.detect_as_dict(blog_text, embedder=models.embedder)

        if enable_logs:
            logger("Step 2/10: Detect constructiveness")
        paragraph_analysis = analyze_paragraphs(
            paragraphs=paragraphs,
            models=models,
            constructiveness_detector=self.constructiveness_detector,
        )

        if enable_logs:
            logger("Step 3/10: Detect toxicity")
            logger("Step 4/10: Detect reasoning markers")
            logger("Step 5/10: Detect explanation depth")
            logger("Step 6/10: Detect writer DNA")
        preliminary_dna = self.writer_dna_classifier.classify(blog_text, embedder=None)
        dna_result = self.writer_dna_classifier.classify(blog_text, embedder=models.embedder)
        if preliminary_dna.writer_type != dna_result.writer_type:
            dna_result.writer_type_probabilities = {
                **dna_result.writer_type_probabilities,
                "preliminary_rule_type": preliminary_dna.writer_type,
            }

        if enable_logs:
            logger("Step 7/10: Apply stance-aware weighting")
        stance_weighted_analysis = apply_stance_aware_weighting(paragraph_analysis, stance_result)

        if enable_logs:
            logger("Step 8/10: Apply adaptive weighting")
        adaptive_result = apply_adaptive_scoring(
            paragraph_analysis=stance_weighted_analysis,
            writer_type=dna_result.writer_type,
            writer_type_probabilities=dna_result.writer_type_probabilities,
        )

        if enable_logs:
            logger("Step 9/10: Apply toxicity override scaling")
        result = apply_toxicity_penalty(adaptive_result, writer_type=dna_result.writer_type)

        stats_verification = verify_blog_statistics(blog_text, writer_type=dna_result.writer_type)
        stats_adjustment = float(stats_verification.get("stat_score_adjustment", 0.0))
        if stats_verification.get("stats_found", False):
            adjusted_score = float(result.get("final_score", 0.0)) + stats_adjustment
            adjusted_score = max(20.0, min(95.0, adjusted_score))
            result["final_score"] = round(adjusted_score, 2)
            result["score_out_of_100"] = result["final_score"]
            component_scores = dict(result.get("component_scores", {}))
            component_scores["stat_accuracy_component"] = round(stats_adjustment, 2)
            result["component_scores"] = component_scores

        writing_quality = adaptive_result.get("writing_quality", {}) if isinstance(adaptive_result.get("writing_quality", {}), dict) else {}
        writing_quality_aggregate = writing_quality.get("aggregate", {}) if isinstance(writing_quality.get("aggregate", {}), dict) else {}
        writing_quality_component = compute_writing_quality_component(writing_quality_aggregate)
        stance_label = str(stance_result.get("stance_label", "NEUTRAL_ANALYSIS"))
        if stance_label in {"DIRECT_ATTACK", "DISMISSIVE_COMPLAINT"}:
            writing_quality_component = min(writing_quality_component, 3.0)

        base_score = float(result.get("final_score", 0.0))
        score_with_quality = max(20.0, min(95.0, base_score + writing_quality_component))
        result["final_score"] = round(score_with_quality, 2)
        result["score_out_of_100"] = result["final_score"]

        component_scores = dict(result.get("component_scores", {}))
        component_scores["writing_quality_component"] = round(writing_quality_component, 2)
        result["component_scores"] = component_scores

        print(
            "WRITING QUALITY DEBUG:",
            round(float(writing_quality_aggregate.get("coherence_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("lexical_diversity_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("sentence_variety_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("repetition_penalty", 0.0)), 4),
            round(float(writing_quality_aggregate.get("position_clarity_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("counter_argument_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("evidence_presence_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("completeness_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("information_density_score", 0.0)), 4),
            round(float(writing_quality_aggregate.get("argument_logic_score", 0.0)), 4),
            round(float(writing_quality_component), 4),
        )

        result["writing_quality_breakdown"] = {
            "coherence_score": round(float(writing_quality_aggregate.get("coherence_score", 0.0)), 4),
            "lexical_diversity_score": round(float(writing_quality_aggregate.get("lexical_diversity_score", 0.0)), 4),
            "sentence_variety_score": round(float(writing_quality_aggregate.get("sentence_variety_score", 0.0)), 4),
            "repetition_penalty": round(float(writing_quality_aggregate.get("repetition_penalty", 0.0)), 4),
            "position_clarity_score": round(float(writing_quality_aggregate.get("position_clarity_score", 0.0)), 4),
            "counter_argument_score": round(float(writing_quality_aggregate.get("counter_argument_score", 0.0)), 4),
            "evidence_presence_score": round(float(writing_quality_aggregate.get("evidence_presence_score", 0.0)), 4),
            "completeness_score": round(float(writing_quality_aggregate.get("completeness_score", 0.0)), 4),
            "information_density_score": round(float(writing_quality_aggregate.get("information_density_score", 0.0)), 4),
            "argument_logic_score": round(float(writing_quality_aggregate.get("argument_logic_score", 0.0)), 4),
            "aggregate_component": round(float(writing_quality_component), 4),
        }
        result["writing_quality_paragraphs"] = writing_quality.get("paragraphs", [])

        result = apply_stance_score_calibration(result)

        bqs_aggregation = aggregate_bqs(
            stance_result=stance_result,
            stats_verification=stats_verification,
            writing_quality_breakdown=result.get("writing_quality_breakdown", {}),
            component_scores=result.get("component_scores", {}),
            writer_type_probabilities=dna_result.writer_type_probabilities,
            signals=result.get("signals", {}),
            blog_text=blog_text,
        )
        result["bqs_aggregation"] = bqs_aggregation
        result["final_score"] = float(bqs_aggregation.get("final_bqs_score", result.get("final_score", 20.0)))
        result["score_out_of_100"] = result["final_score"]

        if enable_logs:
            logger("Step 10/10: Generate final score")

        result["stance_label"] = stance_result.get("stance_label", "NEUTRAL_ANALYSIS")
        result["stance_confidence"] = stance_result.get("stance_confidence", 0.0)
        result["stance_probabilities"] = stance_result.get("stance_probabilities", {})
        result["primary_stance_label"] = stance_result.get("primary_stance_label", "NEUTRAL_COMMENTARY")
        result["primary_stance_confidence"] = stance_result.get("primary_stance_confidence", 0.0)
        result["primary_stance_probabilities"] = stance_result.get("primary_stance_probabilities", {})
        result["style_tags"] = stance_result.get("style_tags", [])
        result["scalar_metrics"] = stance_result.get("scalar_metrics", {})
        result["sarcasm_gate_reason"] = stance_result.get("sarcasm_gate_reason", "")
        result["supportive_defense_strength"] = stance_result.get("supportive_defense_strength", 0.0)
        result["fairness_defense_score"] = stance_result.get("fairness_defense_score", 0.0)
        result["reputation_defense_score"] = stance_result.get("reputation_defense_score", 0.0)
        result["criticism_reference_score"] = stance_result.get("criticism_reference_score", 0.0)
        result["context_change_score"] = stance_result.get("context_change_score", 0.0)
        result["evaluation_redirection_score"] = stance_result.get("evaluation_redirection_score", 0.0)
        result["credibility_restoration_score"] = stance_result.get("credibility_restoration_score", 0.0)
        result["contrast_structure_score"] = stance_result.get("contrast_structure_score", 0.0)
        result["causal_defense_score"] = stance_result.get("causal_defense_score", 0.0)
        result["credibility_defense_score"] = stance_result.get("credibility_defense_score", 0.0)
        result["contrast_rejection_detected"] = stance_result.get("contrast_rejection_detected", False)
        result["quoted_attack_detected"] = stance_result.get("quoted_attack_detected", False)
        result["attack_endorsement_detected"] = stance_result.get("attack_endorsement_detected", False)
        result["attack_rejection_detected"] = stance_result.get("attack_rejection_detected", False)
        result["paragraph_stances"] = stance_result.get("paragraph_stances", [])
        result["overall_stance"] = stance_result.get("overall_stance", {})
        result["stats_verification"] = stats_verification
        result["archetype_detected"] = bqs_aggregation.get("archetype_detected", "analyst_score")

        result["meta"] = {
            "paragraph_count": len(paragraphs),
            "sentence_count": sum(len(p.sentences) for p in paragraphs),
            "device": models.device,
            "models": {
                "embedding": ModelLoader.EMBEDDING_MODEL,
                "toxicity": ModelLoader.TOXICITY_MODEL,
                "emotion": ModelLoader.EMOTION_MODEL,
                "sentiment": ModelLoader.SENTIMENT_MODEL,
                "constructiveness_detector": ConstructivenessDetector.MODEL_NAME,
            },
            "writer_dna_signals": {
                "rule_scores": dna_result.rule_scores,
                "embedding_scores": dna_result.embedding_scores,
            },
            "calibration_mode": "writer-aware scoring, not generic sentiment polarity",
        }
        return result
