from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    text: str
    expected_writer_type: str
    expected_score_range: Tuple[int, int]
    toxicity_expectation: str
    explanation: str


def _fan_samples() -> List[CalibrationSample]:
    rows = [
        ("F01", "I am frustrated with how we closed the innings, but I still back this team fully. If we rotate the strike earlier and use a calmer middle-over plan, we can recover quickly in the next match.", (78, 88)),
        ("F02", "We deserved better today, yet the answer is not panic. I want the captain to trust the in-form bowler for one extra over and tighten the field on the leg side.", (77, 87)),
        ("F03", "I care deeply about this squad and that is why I am being direct. Our powerplay intent is good, but we should leave fewer risky drives when two wickets fall early.", (76, 86)),
        ("F04", "This loss hurts, but the players are trying and deserve fair support. If we improve shot selection after drinks and communicate better between wickets, results will turn.", (78, 89)),
        ("F05", "As a fan, I am emotional after that collapse, yet I can see a path forward. The batting group should commit to singles for five overs before attacking again.", (75, 86)),
        ("F06", "I am disappointed, not hopeless, because this team has talent. A clearer bowling plan at the death and smarter review usage would help us close tight games.", (76, 87)),
        ("F07", "We all felt the pressure in the final overs, but blaming one player is unfair. Let us fix boundary options and support the young finisher with a stable partner.", (77, 88)),
        ("F08", "I love this team enough to criticize it honestly. Our fielding energy dipped late, so the staff should rehearse pressure catches and relay throws before the next game.", (78, 89)),
        ("F09", "I am upset because we keep repeating the same mistake after early momentum. Please set a simple chase template and commit to disciplined running between wickets.", (76, 86)),
        ("F10", "We can bounce back if we stay united and practical. I want more intent in strike rotation and better communication so panic does not spread after one bad over.", (77, 87)),
        ("F11", "This was a painful night, but our supporters should push for improvement, not abuse. If we choose matchups earlier and keep one anchoring batter at all times, we improve quickly.", (79, 90)),
        ("F12", "I am angry at the result, yet I still trust the group. We should protect the left-right pair in the middle overs and stop gifting soft dismissals to part-time spin.", (76, 87)),
        ("F13", "We did many things right before losing control in ten minutes. The fix is patience and clearer on-field calls, not dramatic changes to the entire lineup.", (77, 88)),
        ("F14", "I support this side every week, so this criticism comes from commitment. Bowl fuller at the death and guard square boundaries first, then force batters to hit straight.", (78, 89)),
        ("F15", "I am disappointed by the finish, but this team is close to turning the corner. Keep the tempo steady, reward effort, and reduce reckless shots after wickets.", (77, 88)),
    ]
    return [
        CalibrationSample(
            sample_id=sid,
            text=text,
            expected_writer_type="Passionate Fan",
            expected_score_range=score_range,
            toxicity_expectation="low",
            explanation="Respectful fan voice with constructive, team-focused improvement suggestions.",
        )
        for sid, text, score_range in rows
    ]


def _analyst_samples() -> List[CalibrationSample]:
    rows = [
        ("A01", "The chase failed because the middle-over intent collapsed after the first wicket. A better sequence is rotating strike for four overs, forcing boundary riders in, then targeting the fifth bowler.", (82, 94)),
        ("A02", "Our powerplay looked efficient, yet the transition phase lacked a clear trigger. The batting unit should predefine risk windows based on matchup quality rather than reacting ball by ball.", (81, 93)),
        ("A03", "The death overs unraveled due to predictable pace lengths and delayed field changes. Using a hard-length option with a deep square set would reduce the high-percentage slog areas.", (83, 95)),
        ("A04", "The lineup structure is not broken, but role sequencing is. Sending the left-hander earlier against off-spin would reduce dot pressure and preserve finishing resources.", (80, 92)),
        ("A05", "The bowling strategy under pressure lacked contingency planning. If yorker execution drops, the immediate shift should be wide pace-off lines with boundary-protection geometry.", (82, 94)),
        ("A06", "The team created momentum, then leaked control through poor phase management. The fix is a clear control metric: limit high-risk balls per over before accelerating.", (81, 93)),
        ("A07", "Field settings were one decision late throughout the final spell. A proactive leg-side trap against the set batter would have reduced boundary frequency immediately.", (82, 94)),
        ("A08", "The collapse was less about talent and more about decision hierarchy. Shot selection should be anchored to game state, not individual confidence after a single boundary.", (80, 92)),
        ("A09", "Our intent profile was volatile across phases, producing avoidable pressure spikes. Stabilizing through controlled singles first would create better launch conditions at the end.", (81, 93)),
        ("A10", "The spin matchup was misread after the break. A simple adjustment, attacking with the sweep line then rotating behind square, would have neutralized that plan.", (82, 94)),
        ("A11", "Review decisions reflected emotion rather than expected value. Waiting one extra second for keeper and slip confirmation would materially improve review efficiency.", (80, 92)),
        ("A12", "The key issue is process consistency under pressure. Tactical clarity, especially role communication before each phase, would reduce panic-driven execution errors.", (82, 94)),
        ("A13", "The team defended well until lengths drifted into hitting arcs. Protecting straight boundaries and mixing pace bands earlier would have constrained clean swings.", (81, 93)),
        ("A14", "Selection should prioritize phase complementarity, not just individual form. The current combination leaves a control gap between overs twelve and sixteen.", (80, 92)),
        ("A15", "There is a workable blueprint here, but adaptation speed is lagging. Faster tactical pivots against set batters would convert close games into wins.", (81, 93)),
    ]
    return [
        CalibrationSample(
            sample_id=sid,
            text=text,
            expected_writer_type="Analyst",
            expected_score_range=score_range,
            toxicity_expectation="low",
            explanation="Reasoned, structured cricket analysis with causal logic and tactical recommendations.",
        )
        for sid, text, score_range in rows
    ]


def _story_samples() -> List[CalibrationSample]:
    rows = [
        ("S01", "The stadium held its breath when our young finisher walked in, bat tucked under his arm. He looked nervous at first, then settled, and even though we lost, his fight gave the crowd hope.", (76, 88)),
        ("S02", "When the lights flickered over the outfield, the match felt like a film scene. We stumbled late, but the captain gathered everyone calmly, and that composure mattered more than the scoreline.", (75, 87)),
        ("S03", "I still remember the silence after that dropped catch. Yet moments later, the same fielder chased down a boundary and lifted the team again.", (75, 86)),
        ("S04", "The innings began with chaos and ended with a lesson in patience. We were emotional in the stands, but the players showed grit that deserves respect.", (76, 88)),
        ("S05", "Rain hovered, drums echoed, and the chase kept twisting with every over. We fell short, but the story was about resilience, not collapse.", (75, 87)),
        ("S06", "At drinks, the dugout looked tense, yet one senior player kept speaking calmly to the younger batters. That quiet leadership steadied the chase for a while.", (76, 88)),
        ("S07", "The final over felt endless as every ball changed the mood in the stands. We lost, but the team never stopped fighting for each run.", (75, 87)),
        ("S08", "He walked back slowly after edging behind, helmet under his arm, while the crowd applauded anyway. It was a loss, but also a reminder of commitment and character.", (76, 88)),
        ("S09", "Our supporters sang through the pressure, and the players responded with heart. The result hurt, yet the performance still felt meaningful and alive.", (75, 87)),
        ("S10", "The match turned in a single spell that felt like thunder. Even then, the side kept believing, and that belief is why this team remains easy to love.", (76, 88)),
        ("S11", "From the first roar to the final silence, the game moved like a story of near misses and brave moments. We did not finish strong, but we never looked indifferent.", (76, 88)),
        ("S12", "I watched a rookie bowler hide his nerves behind a quick smile before every run-up. He got hit, adjusted, and came back better by the end.", (75, 87)),
        ("S13", "The chase drifted, recovered, and drifted again like a tide. We left with disappointment, but also with faith that this group is still growing together.", (75, 87)),
        ("S14", "In the noise of a close defeat, one image stayed with me: teammates waiting at the rope to encourage the next batter. That unity matters in long seasons.", (76, 88)),
        ("S15", "When the last wicket fell, no one moved for a moment. Then applause rose, because effort and courage were visible even in defeat.", (75, 87)),
    ]
    return [
        CalibrationSample(
            sample_id=sid,
            text=text,
            expected_writer_type="Storyteller",
            expected_score_range=score_range,
            toxicity_expectation="low",
            explanation="Narrative, coherent cricket storytelling with emotional flow and respectful tone.",
        )
        for sid, text, score_range in rows
    ]


def _debate_samples() -> List[CalibrationSample]:
    rows = [
        ("D01", "The captain deserves criticism for late bowling changes, but dropping him would be excessive. A fair view is to keep him while adding tactical support in high-pressure phases.", (82, 95)),
        ("D02", "Some argue the batting order should stay fixed, while others want flexibility by matchup. The balanced approach is a stable top order with situational middle-order swaps.", (81, 94)),
        ("D03", "It is valid to blame execution, yet planning also failed in the final overs. Both can be true, and improvement requires accountability at both levels.", (82, 95)),
        ("D04", "Fans calling for wholesale changes are reacting to pain, but continuity has value too. Keep the core, adjust roles, and evaluate outcomes over several matches.", (80, 93)),
        ("D05", "The opener played too slowly, but the middle order also failed to rotate under pressure. Critique should be distributed fairly rather than pinned on one player.", (81, 94)),
        ("D06", "One side says we lacked intent; the other says we lacked control. The stronger argument is that intent without control creates collapse in tight chases.", (82, 95)),
        ("D07", "Selection debates often ignore role fit. A popular player may still be the wrong option for a specific matchup, so decisions should balance merit and context.", (81, 94)),
        ("D08", "It is reasonable to demand better death bowling, but calling the unit hopeless is unfair. The evidence suggests targeted role clarity would solve most of the issue.", (82, 95)),
        ("D09", "Some supporters want immediate youth promotion, whereas others prioritize experienced control. A balanced policy introduces youth gradually inside clear tactical roles.", (80, 93)),
        ("D10", "The coaching plan should be questioned, though blame should not become personal abuse. Criticism is strongest when it includes alternatives and shared accountability.", (82, 95)),
        ("D11", "You can defend the captain's long-term record and still challenge this match's decisions. Debate should compare options, not silence one side.", (81, 94)),
        ("D12", "The lineup has flaws, yet panic replacements could worsen role confusion. Rational debate points to incremental adjustment instead of emotional overhaul.", (80, 93)),
        ("D13", "Aggressive intent can win matches, but blind aggression breaks chases. The better position is controlled aggression with clear risk windows.", (82, 95)),
        ("D14", "Critics are right that standards slipped, while defenders are right that the squad has shown resilience. Productive debate blends both truths into practical next steps.", (81, 94)),
        ("D15", "It is fair to question leadership under pressure, but unfair to attack character. Keep the argument on decisions, evidence, and improvement pathways.", (82, 95)),
    ]
    return [
        CalibrationSample(
            sample_id=sid,
            text=text,
            expected_writer_type="Debater",
            expected_score_range=score_range,
            toxicity_expectation="low",
            explanation="Balanced argument structure with opposing viewpoints and constructive resolution.",
        )
        for sid, text, score_range in rows
    ]


def get_calibration_samples() -> List[CalibrationSample]:
    return _fan_samples() + _analyst_samples() + _story_samples() + _debate_samples()
