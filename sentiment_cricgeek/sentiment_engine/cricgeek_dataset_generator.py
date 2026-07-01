from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple


def _word_count(text: str) -> int:
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


def _build_paragraph(seed: int, sentence_banks: List[List[str]], min_words: int = 70, max_words: int = 150) -> str:
    rng = random.Random(seed)
    sentences: List[str] = []

    for bank in sentence_banks:
        sentences.append(rng.choice(bank))

    rng.shuffle(sentences)
    paragraph = " ".join(sentences)

    # Expand until minimum length.
    while _word_count(paragraph) < min_words:
        source_bank = rng.choice(sentence_banks)
        paragraph = paragraph + " " + rng.choice(source_bank)

    # Trim if too long by dropping the last sentence(s).
    while _word_count(paragraph) > max_words and len(sentences) > 3:
        sentences = sentences[:-1]
        paragraph = " ".join(sentences)

    if _word_count(paragraph) > max_words:
        words = paragraph.split()
        paragraph = " ".join(words[:max_words])

    return paragraph.strip()


def _template_bank() -> Dict[str, List[List[str]]]:
    fan_open = [
        "I walked out of the ground frustrated, but I still believe this side is close to a strong run.",
        "As a long-time supporter, this loss stings, yet my faith in the squad is still firm.",
        "I am emotional after that finish, but my criticism comes from loyalty, not anger.",
        "This result hurt, though I still back the team because the effort was visible.",
    ]
    fan_middle = [
        "If we rotate strike earlier after a wicket, the chase does not freeze in the middle overs.",
        "The captain should trust matchup bowling sooner, especially when the set batter targets one side.",
        "Our fielders need a clearer call system at the rope so pressure moments do not become chaos.",
        "The batting unit can improve by protecting one anchor while hitters attack in pairs.",
    ]
    fan_close = [
        "That adjustment alone could turn close defeats into stable wins across a long series.",
        "I am asking for practical changes, not panic changes, because the core quality is already present.",
        "With calmer decisions and better role discipline, this group can recover quickly.",
        "Supporters should demand smarter execution while still backing every player to improve.",
    ]

    analyst_open = [
        "The chase broke down because decision velocity dropped after the first pressure over.",
        "The innings pattern shows a tactical gap between consolidation and acceleration phases.",
        "The bowling control problem came from predictable release points in back-to-back overs.",
        "This match exposed sequencing issues more than talent limitations.",
    ]
    analyst_middle = [
        "A better approach is to predefine risk windows by matchup and protect low-risk singles between them.",
        "When two right-handers are set, the captain should shift square protection before pace-off variations begin.",
        "Selection can improve by pairing one control bowler with one strike bowler in the same phase, not separate phases.",
        "The middle order should carry role-based triggers so tempo choices are not improvised ball by ball.",
    ]
    analyst_close = [
        "That structure would reduce volatility and improve conversion in one-score games.",
        "This is a process correction problem, so incremental tactical fixes are preferable to wholesale changes.",
        "If these adjustments are implemented, expected outcomes should stabilize over the next block of fixtures.",
        "The evidence suggests role clarity and sequencing discipline are the highest-value corrections.",
    ]

    story_open = [
        "When the lights came on for the final session, the stadium sounded like one loud heartbeat.",
        "The game kept twisting, and every over seemed to carry a different emotion across the stands.",
        "I remember the silence after the dropped chance, then the roar when the next boundary was saved.",
        "The evening felt dramatic from the first ball, with momentum changing every few minutes.",
    ]
    story_middle = [
        "A young batter looked uncertain early, then settled by choosing patient singles before expanding his range.",
        "The senior bowler missed his first mark, reset his field, and returned with better control under pressure.",
        "Even after a collapse, teammates met each incoming player at the rope with calm and clear encouragement.",
        "The captain kept speaking between deliveries, and that steady voice prevented visible panic.",
    ]
    story_close = [
        "We still lost, yet the story was about resilience and adjustment rather than surrender.",
        "That night ended in disappointment, but it also revealed why this group remains easy to trust.",
        "The scoreboard hurt, though the character shown in hard moments gave supporters real hope.",
        "By the final whistle, the result was harsh, but the performance had genuine heart and direction.",
    ]

    debater_open = [
        "One side argues the captain reacted too late, while another side says execution let him down first.",
        "Some fans want immediate selection changes, but others warn that panic selection can break role balance.",
        "It is fair to question tactical calls, yet unfair to ignore the pressure context players were handling.",
        "There are two credible readings of this defeat, and both deserve serious examination.",
    ]
    debater_middle = [
        "A balanced view is to keep the core lineup while redefining death-over responsibilities by matchup.",
        "Criticism should focus on decision quality and alternatives, not on personal character judgments.",
        "The stronger argument combines accountability with continuity, because both stability and adaptation matter.",
        "Comparing both viewpoints shows that role discipline, not dramatic overhaul, is the practical middle ground.",
    ]
    debater_close = [
        "That synthesis respects legitimate concerns while keeping the solution actionable.",
        "Debate is most useful when it compares evidence and ends in concrete next steps.",
        "This framing allows rigorous criticism without slipping into unfair blame language.",
        "If discussion stays balanced, the team can improve without losing internal confidence.",
    ]

    complaint_open = [
        "I am tired of watching the same script every week and nothing ever changes.",
        "This was another disappointing match and the whole thing felt flat from start to finish.",
        "The team looked ordinary again, and the night became frustrating very quickly.",
        "I cannot enjoy these games lately because the pattern feels repetitive and draining.",
    ]
    complaint_middle = [
        "The batting never looked convincing and the bowling followed the same old mistakes.",
        "Every phase felt messy, and no part of the performance looked genuinely stable.",
        "The effort seemed inconsistent, then momentum disappeared for long stretches.",
        "The crowd kept waiting for control, but the performance stayed uneven throughout.",
    ]
    complaint_close = [
        "I wish things were better, but right now it just feels like more of the same.",
        "There was no clear improvement tonight, only another frustrating ending.",
        "Supporters deserve stronger performances than this recurring pattern.",
        "At the moment I mostly feel disappointed rather than optimistic about the next fixture.",
    ]

    toxic_open = [
        "The captain is a complete idiot who has no business leading this team.",
        "That batter is useless and should be thrown out immediately.",
        "These players are pathetic clowns who embarrass the club every week.",
        "The coaching staff are incompetent fools with zero cricket intelligence.",
    ]
    toxic_middle = [
        "They choke under pressure because they are mentally weak and clueless.",
        "Every decision proves they are hopeless and not fit for professional sport.",
        "The lineup is full of losers who cannot do even basic things right.",
        "No serious team would trust such garbage-level competence in key moments.",
    ]
    toxic_close = [
        "Anyone defending them is blind to how awful and worthless this group has become.",
        "They deserve to be ridiculed until they disappear from the tournament.",
        "This squad is a joke and every one of them should be ashamed.",
        "Nothing can fix this mess because they are fundamentally incapable.",
    ]

    return {
        "Constructive Passionate Fan": [fan_open, fan_middle, fan_close, fan_middle, fan_close],
        "Constructive Analyst": [analyst_open, analyst_middle, analyst_close, analyst_middle, analyst_close],
        "Constructive Storyteller": [story_open, story_middle, story_close, story_middle, story_close],
        "Constructive Debater": [debater_open, debater_middle, debater_close, debater_middle, debater_close],
        "Non-constructive complaint": [complaint_open, complaint_middle, complaint_close, complaint_middle, complaint_close],
        "Toxic competence-attack writing": [toxic_open, toxic_middle, toxic_close, toxic_middle, toxic_close],
    }


def _category_metadata(category: str) -> Tuple[str, bool, bool, List[int], str]:
    if category == "Constructive Passionate Fan":
        return ("Passionate Fan", True, False, [75, 90], "Respectful fan emotion with actionable cricket suggestions and team-focused intent.")
    if category == "Constructive Analyst":
        return ("Analyst", True, False, [80, 95], "Cause-effect cricket reasoning with role-based tactical analysis and clear recommendations.")
    if category == "Constructive Storyteller":
        return ("Storyteller", True, False, [75, 90], "Narrative match storytelling with coherent flow, emotional depth, and constructive reflection.")
    if category == "Constructive Debater":
        return ("Debater", True, False, [80, 95], "Balanced argument structure comparing viewpoints and landing on practical next steps.")
    if category == "Non-constructive complaint":
        return ("All-Rounder", False, False, [40, 60], "Complaint-heavy writing with little explanation or actionable guidance.")
    return ("All-Rounder", False, True, [10, 35], "Abusive competence attacks and degrading language that should trigger strong toxicity penalties.")


def generate_cricgeek_dataset(total_per_category: int = 20) -> List[Dict]:
    categories = [
        "Constructive Passionate Fan",
        "Constructive Analyst",
        "Constructive Storyteller",
        "Constructive Debater",
        "Non-constructive complaint",
        "Toxic competence-attack writing",
    ]
    banks = _template_bank()

    dataset: List[Dict] = []
    sample_id = 1

    for category_idx, category in enumerate(categories):
        for item_idx in range(total_per_category):
            paragraph = _build_paragraph(
                seed=137 + category_idx * 1000 + item_idx * 17,
                sentence_banks=banks[category],
                min_words=70,
                max_words=150,
            )

            expected_writer_type, expected_constructive, expected_toxic, expected_score_range, explanation = _category_metadata(category)

            dataset.append(
                {
                    "id": sample_id,
                    "category": category,
                    "text": paragraph,
                    "expected_writer_type": expected_writer_type,
                    "expected_constructive": expected_constructive,
                    "expected_toxic": expected_toxic,
                    "expected_score_range": expected_score_range,
                    "explanation": explanation,
                    "word_count": _word_count(paragraph),
                }
            )
            sample_id += 1

    return dataset


def save_dataset(file_path: str = "cricgeek_constructiveness_dataset.json") -> List[Dict]:
    data = generate_cricgeek_dataset(total_per_category=20)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def main() -> None:
    data = save_dataset()
    print(f"Generated {len(data)} calibration paragraphs at cricgeek_constructiveness_dataset.json")


if __name__ == "__main__":
    main()
