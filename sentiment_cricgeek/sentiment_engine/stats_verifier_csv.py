from __future__ import annotations

import csv
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "t20s_csv"
CACHE_PATH = Path(__file__).resolve().parents[1] / "t20s_dataset_cache.pkl"
CACHE_VERSION = 2

STAT_TYPES = {
    "batting_average": ["average", "batting average"],
    "strike_rate": ["strike rate", "sr"],
    "economy_rate": ["economy", "economy rate"],
    "runs_total": ["runs", "run"],
    "wickets_total": ["wickets", "wicket"],
}

WINDOW_TYPES = {
    "last_n_matches": r"last\s+(\d{1,3})\s+matches?",
    "last_n_innings": r"last\s+(\d{1,3})\s+innings?",
}

YEAR_RANGE_PATTERN = re.compile(r"(?:between|from)\s+(20\d{2})\s+(?:and|to)\s+(20\d{2})", re.IGNORECASE)
SEASON_PATTERN = re.compile(r"(?:this|current)\s+season", re.IGNORECASE)
VS_PATTERN = re.compile(r"\b(?:vs|against)\s+([A-Z][A-Za-z\.\-\s]{1,40})")
AT_PATTERN = re.compile(r"\bat\s+([A-Z][A-Za-z\.\-\s]{2,60})")
CLAIMED_VALUE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")

NON_BOWLER_WICKETS = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "handled the ball",
    "timed out",
}


_DATASET_CACHE: Optional[pd.DataFrame] = None
_PLAYER_NAMES_CACHE: Optional[List[str]] = None
_PLAYER_NAME_LOOKUP_CACHE: Optional[Dict[str, str]] = None


@dataclass
class ParsedClaim:
    stat_type: str
    window_type: Optional[str]
    window_size: Optional[int]
    opposition: Optional[str]
    venue: Optional[str]
    season: Optional[str]
    year_range: Optional[Tuple[int, int]]
    claimed_value: Optional[float]
    sentence: str


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _normalize_name(name: str) -> str:
    compact = re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()
    compact = re.sub(r"\s+", " ", compact)
    return compact


def _normalize_team(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _extract_year(date_text: str) -> Optional[int]:
    if not date_text:
        return None
    match = re.search(r"(20\d{2})", date_text)
    if not match:
        return None
    return int(match.group(1))


def _season_to_year(season_text: str, match_year: Optional[int]) -> Optional[str]:
    if season_text:
        return season_text
    if match_year is None:
        return None
    return str(match_year)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _name_variants(name: str) -> List[str]:
    variants = {name}
    stripped = re.sub(r"\.", "", name)
    variants.add(stripped)
    variants.add(_normalize_name(name))
    parts = name.split()
    if len(parts) >= 2:
        variants.add(parts[-1])
    return [v for v in variants if v]


def _extract_stat_type(sentence: str) -> Optional[str]:
    lowered = sentence.lower()
    if "strike rate" in lowered or re.search(r"\bsr\b", lowered):
        return "strike_rate"
    if "economy" in lowered:
        return "economy_rate"
    if "average" in lowered:
        return "batting_average"
    if "wicket" in lowered:
        return "wickets_total"
    if "run" in lowered:
        return "runs_total"
    return None


def _extract_window(sentence: str) -> Tuple[Optional[str], Optional[int]]:
    for window_name, pattern in WINDOW_TYPES.items():
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return window_name, int(match.group(1))
    return None, None


def _extract_year_range(sentence: str) -> Optional[Tuple[int, int]]:
    match = YEAR_RANGE_PATTERN.search(sentence)
    if not match:
        return None
    start_year = int(match.group(1))
    end_year = int(match.group(2))
    if end_year < start_year:
        start_year, end_year = end_year, start_year
    return start_year, end_year


def _extract_opposition(sentence: str) -> Optional[str]:
    match = VS_PATTERN.search(sentence)
    if not match:
        return None
    return _normalize_team(match.group(1))


def _extract_venue(sentence: str) -> Optional[str]:
    match = AT_PATTERN.search(sentence)
    if not match:
        return None
    candidate = _normalize_team(match.group(1))
    if candidate.lower().startswith("the "):
        candidate = candidate[4:]
    return candidate


def load_t20_dataset() -> pd.DataFrame:
    global _DATASET_CACHE

    if _DATASET_CACHE is not None:
        return _DATASET_CACHE

    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as handle:
                cached = pickle.load(handle)

            # Legacy cache format: raw DataFrame.
            if isinstance(cached, pd.DataFrame):
                _DATASET_CACHE = cached
                return _DATASET_CACHE

            # Versioned cache format.
            if isinstance(cached, dict):
                cache_version = int(cached.get("cache_version", 0))
                cached_df = cached.get("dataframe")
                if cache_version == CACHE_VERSION and isinstance(cached_df, pd.DataFrame):
                    _DATASET_CACHE = cached_df
                    return _DATASET_CACHE
        except Exception:
            # Incompatible/stale pickle (e.g. pandas dtype ABI mismatch): rebuild safely.
            try:
                CACHE_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    rows: List[Dict[str, Any]] = []
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    for file_path in csv_files:
        match_id = file_path.stem
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)

            teams: List[str] = []
            player_team: Dict[str, str] = {}
            match_date = ""
            season = ""
            venue = ""

            batting_stats: Dict[str, Dict[str, float]] = {}
            bowling_stats: Dict[str, Dict[str, float]] = {}
            dismissals: Dict[str, int] = {}

            for raw in reader:
                if not raw:
                    continue
                row = list(raw) + [""] * max(0, 16 - len(raw))
                row_type = row[0]

                if row_type == "info":
                    key = row[1]
                    if key == "team":
                        teams.append(_normalize_team(row[2]))
                    elif key == "date":
                        match_date = row[2]
                    elif key == "season":
                        season = row[2]
                    elif key == "venue":
                        venue = row[2]
                    elif key == "player" and len(row) >= 4:
                        team_name = _normalize_team(row[2])
                        player_name = row[3].strip()
                        player_team[player_name] = team_name
                    continue

                if row_type != "ball":
                    continue

                striker = row[4].strip()
                bowler = row[6].strip()
                batsman_runs = _to_int(row[7])
                extras_total = _to_int(row[8])
                wides = _to_int(row[9])
                noballs = _to_int(row[10])
                byes = _to_int(row[11])
                legbyes = _to_int(row[12])
                penalty = _to_int(row[13])
                wicket_type = row[14].strip().lower()
                player_dismissed = row[15].strip()

                legal_delivery = (wides == 0 and noballs == 0)

                batting_entry = batting_stats.setdefault(
                    striker,
                    {
                        "runs": 0.0,
                        "balls": 0.0,
                    },
                )
                batting_entry["runs"] += batsman_runs
                if legal_delivery:
                    batting_entry["balls"] += 1.0

                bowling_entry = bowling_stats.setdefault(
                    bowler,
                    {
                        "runs_conceded": 0.0,
                        "balls_bowled": 0.0,
                        "wickets": 0.0,
                    },
                )
                bowler_extras = wides + noballs + penalty
                bowling_entry["runs_conceded"] += batsman_runs + bowler_extras
                if legal_delivery:
                    bowling_entry["balls_bowled"] += 1.0

                if player_dismissed:
                    if wicket_type not in {"retired hurt"}:
                        dismissals[player_dismissed] = dismissals.get(player_dismissed, 0) + 1
                    if wicket_type and wicket_type not in NON_BOWLER_WICKETS:
                        bowling_entry["wickets"] += 1.0

            all_players = set(player_team.keys()) | set(batting_stats.keys()) | set(bowling_stats.keys())
            team_a = teams[0] if len(teams) > 0 else ""
            team_b = teams[1] if len(teams) > 1 else ""
            match_year = _extract_year(match_date)
            season_value = _season_to_year(season, match_year)

            for player in all_players:
                team_name = _normalize_team(player_team.get(player, ""))
                opposition = ""
                if team_name and team_a and team_b:
                    opposition = team_b if team_name == team_a else team_a

                bat = batting_stats.get(player, {"runs": 0.0, "balls": 0.0})
                bowl = bowling_stats.get(player, {"runs_conceded": 0.0, "balls_bowled": 0.0, "wickets": 0.0})

                rows.append(
                    {
                        "match_id": match_id,
                        "match_date": match_date,
                        "year": match_year,
                        "season": season_value,
                        "venue": _normalize_team(venue),
                        "player": player,
                        "player_team": team_name,
                        "opposition": opposition,
                        "runs": float(bat.get("runs", 0.0)),
                        "balls_faced": float(bat.get("balls", 0.0)),
                        "dismissals": float(dismissals.get(player, 0)),
                        "runs_conceded": float(bowl.get("runs_conceded", 0.0)),
                        "balls_bowled": float(bowl.get("balls_bowled", 0.0)),
                        "wickets": float(bowl.get("wickets", 0.0)),
                        "source_file": file_path.name,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.columns = [_normalize_col(col) for col in frame.columns]
    if not frame.empty:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")

    with CACHE_PATH.open("wb") as handle:
        pickle.dump(
            {
                "cache_version": CACHE_VERSION,
                "pandas_version": pd.__version__,
                "dataframe": frame,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    _DATASET_CACHE = frame
    return _DATASET_CACHE


def _player_lookup() -> Dict[str, str]:
    global _PLAYER_NAME_LOOKUP_CACHE, _PLAYER_NAMES_CACHE
    if _PLAYER_NAME_LOOKUP_CACHE is not None and _PLAYER_NAMES_CACHE is not None:
        return _PLAYER_NAME_LOOKUP_CACHE

    frame = load_t20_dataset()
    names = sorted({str(name).strip() for name in frame.get("player", pd.Series(dtype=str)).dropna().tolist() if str(name).strip()})
    lookup: Dict[str, str] = {}
    for name in names:
        for variant in _name_variants(name):
            lookup[_normalize_name(variant)] = name

    _PLAYER_NAMES_CACHE = names
    _PLAYER_NAME_LOOKUP_CACHE = lookup
    return lookup


def extract_player_mentions(text: str) -> List[str]:
    lookup = _player_lookup()
    preprocessed = re.sub(r"\b([A-Za-z\.]+)'s\b", r"\1", text)
    normalized_text = _normalize_name(preprocessed)
    mentions: List[str] = []
    frame = load_t20_dataset()

    initial_surname_map: Dict[Tuple[str, str], List[str]] = {}
    for player_name in frame.get("player", pd.Series(dtype=str)).dropna().astype(str):
        parts = player_name.strip().split()
        if len(parts) < 2:
            continue
        initial = re.sub(r"[^A-Za-z]", "", parts[0]).lower()
        surname = re.sub(r"[^A-Za-z]", "", parts[-1]).lower()
        if not initial or not surname:
            continue
        key = (initial[0], surname)
        initial_surname_map.setdefault(key, []).append(player_name)

    candidates = re.findall(r"[A-Z][A-Za-z\.\-']+(?:\s+[A-Z][A-Za-z\.\-']+){0,3}", preprocessed)
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in lookup:
            canonical = lookup[key]
            if canonical not in mentions:
                mentions.append(canonical)

    initial_surname_candidates = re.findall(r"\b([A-Z])\.?\s+([A-Z][a-zA-Z]+)\b", preprocessed)
    for initial, surname in initial_surname_candidates:
        key = (initial.lower(), surname.lower())
        for candidate in initial_surname_map.get(key, []):
            if candidate not in mentions:
                mentions.append(candidate)

    if mentions:
        return mentions

    for norm_variant, canonical_name in lookup.items():
        if not norm_variant:
            continue
        if re.search(rf"\b{re.escape(norm_variant)}\b", normalized_text):
            if canonical_name not in mentions:
                mentions.append(canonical_name)

    return mentions


def extract_claimed_value(text: str) -> Optional[float]:
    metric_value = re.search(
        r"(?:average|strike\s+rate|economy(?:\s+rate)?|runs?|wickets?)"
        r"[^\d]{0,25}(?:is|are|was|were|at|of)?[^\d]{0,15}(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if metric_value:
        return float(metric_value.group(1))

    numbers = [float(value) for value in CLAIMED_VALUE_PATTERN.findall(text)]
    if not numbers:
        return None

    # In phrases like "last 10 matches average is 32", trailing value is the claim.
    if re.search(r"last\s+\d{1,3}\s+(?:matches?|innings?)", text, re.IGNORECASE):
        return numbers[-1]

    return numbers[0]


def extract_stat_claims(text: str) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    for sentence in sentences:
        clauses = [
            part.strip()
            for part in re.split(r"\b(?:while|however|but)\b|;", sentence, flags=re.IGNORECASE)
            if part.strip()
        ]
        for clause in clauses:
            stat_type = _extract_stat_type(clause)
            if stat_type is None:
                continue

            window_type, window_size = _extract_window(clause)
            year_range = _extract_year_range(clause)
            season = "this_season" if SEASON_PATTERN.search(clause) else None
            opposition = _extract_opposition(clause)
            venue = _extract_venue(clause)
            claimed_value = extract_claimed_value(clause)

            claim = ParsedClaim(
                stat_type=stat_type,
                window_type=window_type,
                window_size=window_size,
                opposition=opposition,
                venue=venue,
                season=season,
                year_range=year_range,
                claimed_value=claimed_value,
                sentence=clause,
            )
            claims.append(
                {
                    "stat_type": claim.stat_type,
                    "window_type": claim.window_type,
                    "window_size": claim.window_size,
                    "opposition": claim.opposition,
                    "venue": claim.venue,
                    "season": claim.season,
                    "year_range": claim.year_range,
                    "claimed_value": claim.claimed_value,
                    "sentence": claim.sentence,
                }
            )

    return claims


def _resolve_opposition(opposition: str) -> str:
    norm = _normalize_team(opposition)
    frame = load_t20_dataset()
    teams = sorted({str(value) for value in frame.get("opposition", pd.Series(dtype=str)).dropna().tolist() if str(value)})
    lower_map = {team.lower(): team for team in teams}
    if norm.lower() in lower_map:
        return lower_map[norm.lower()]

    for team in teams:
        if norm.lower() in team.lower() or team.lower() in norm.lower():
            return team
    return norm


def _resolve_venue(venue: str) -> str:
    norm = _normalize_team(venue)
    frame = load_t20_dataset()
    venues = sorted({str(value) for value in frame.get("venue", pd.Series(dtype=str)).dropna().tolist() if str(value)})
    lower_map = {val.lower(): val for val in venues}
    if norm.lower() in lower_map:
        return lower_map[norm.lower()]

    for candidate in venues:
        if norm.lower() in candidate.lower() or candidate.lower() in norm.lower():
            return candidate
    return norm


def filter_player_rows(player_name: str, filters: Dict[str, Any]) -> pd.DataFrame:
    frame = load_t20_dataset()
    if frame.empty:
        return frame.copy()

    rows = frame[frame["player"] == player_name].copy()

    if filters.get("season"):
        if filters["season"] == "this_season":
            max_season = rows["season"].dropna().astype(str).sort_values().iloc[-1] if not rows["season"].dropna().empty else None
            if max_season is not None:
                rows = rows[rows["season"].astype(str) == str(max_season)]
        else:
            rows = rows[rows["season"].astype(str) == str(filters["season"])]

    if filters.get("year_range"):
        start_year, end_year = filters["year_range"]
        rows = rows[(rows["year"] >= start_year) & (rows["year"] <= end_year)]

    if filters.get("opposition"):
        opposition = _resolve_opposition(str(filters["opposition"]))
        rows = rows[rows["opposition"].astype(str).str.lower() == opposition.lower()]

    if filters.get("venue"):
        venue = _resolve_venue(str(filters["venue"]))
        rows = rows[rows["venue"].astype(str).str.lower() == venue.lower()]

    rows = rows.sort_values("match_date", ascending=False)

    window_type = filters.get("window_type")
    window_size = filters.get("window_size")
    if window_type in {"last_n_matches", "last_n_innings"} and window_size:
        size = max(1, int(window_size))
        rows = rows.head(size)

    return rows


def compute_stat(rows: pd.DataFrame, stat_type: str) -> Optional[float]:
    if rows is None or rows.empty:
        return None

    total_runs = float(rows["runs"].sum())
    total_balls = float(rows["balls_faced"].sum())
    total_dismissals = float(rows["dismissals"].sum())
    total_wickets = float(rows["wickets"].sum())
    total_runs_conceded = float(rows["runs_conceded"].sum())
    total_balls_bowled = float(rows["balls_bowled"].sum())

    if stat_type == "batting_average":
        if total_dismissals <= 0:
            return None
        return total_runs / total_dismissals

    if stat_type == "strike_rate":
        if total_balls <= 0:
            return None
        return (total_runs / total_balls) * 100.0

    if stat_type == "economy_rate":
        overs = total_balls_bowled / 6.0
        if overs <= 0:
            return None
        return total_runs_conceded / overs

    if stat_type == "runs_total":
        return total_runs

    if stat_type == "wickets_total":
        return total_wickets

    return None


def verify_stat_claim(claimed_value: Optional[float], computed_value: Optional[float]) -> Dict[str, Any]:
    if claimed_value is None or computed_value is None:
        return {
            "stats_found": False,
            "stats_verified": False,
            "stat_accuracy_score": 0.0,
            "stat_confidence": "low",
            "difference": None,
        }

    denominator = max(abs(computed_value), 1e-9)
    relative_diff = abs(claimed_value - computed_value) / denominator

    if relative_diff <= 0.05:
        return {
            "stats_found": True,
            "stats_verified": True,
            "stat_accuracy_score": 1.0,
            "stat_confidence": "high",
            "difference": relative_diff,
        }

    if relative_diff <= 0.10:
        return {
            "stats_found": True,
            "stats_verified": False,
            "stat_accuracy_score": 0.6,
            "stat_confidence": "medium",
            "difference": relative_diff,
        }

    return {
        "stats_found": True,
        "stats_verified": False,
        "stat_accuracy_score": 0.0,
        "stat_confidence": "high",
        "difference": relative_diff,
    }


def _writer_stat_weight(writer_type: str, has_numeric_claim: bool) -> float:
    if writer_type == "Analyst":
        return 4.0
    if writer_type == "Debater":
        return 2.5
    if writer_type == "Passionate Fan":
        return 1.5 if has_numeric_claim else 0.25
    if writer_type == "Storyteller":
        return 0.75
    return 1.0


def verify_blog_statistics(text: str, writer_type: str = "All-Rounder") -> Dict[str, Any]:
    mentions = extract_player_mentions(text)
    claims = extract_stat_claims(text)

    stat_mentions: List[Dict[str, Any]] = []

    if not mentions or not claims:
        return {
            "player_mentions": mentions,
            "stat_mentions": stat_mentions,
            "stat_type": None,
            "computed_value": None,
            "claimed_value": None,
            "difference": None,
            "stats_found": False,
            "stats_verified": False,
            "stat_accuracy_score": 0.0,
            "stat_confidence": "low",
            "stat_score_adjustment": 0.0,
        }

    sentence_to_players: Dict[str, List[str]] = {}
    for claim in claims:
        sentence = str(claim.get("sentence", ""))
        players = extract_player_mentions(sentence)
        sentence_to_players[sentence] = players if players else mentions

    for claim in claims:
        sentence = str(claim.get("sentence", ""))
        players = sentence_to_players.get(sentence, mentions)
        filters = {
            "window_type": claim.get("window_type"),
            "window_size": claim.get("window_size"),
            "season": claim.get("season"),
            "year_range": claim.get("year_range"),
            "opposition": claim.get("opposition"),
            "venue": claim.get("venue"),
        }

        for player in players:
            rows = filter_player_rows(player, filters)
            computed = compute_stat(rows, str(claim.get("stat_type")))
            verification = verify_stat_claim(claim.get("claimed_value"), computed)

            stat_mentions.append(
                {
                    "player": player,
                    "stat_type": claim.get("stat_type"),
                    "window_type": claim.get("window_type"),
                    "window_size": claim.get("window_size"),
                    "opposition": claim.get("opposition"),
                    "venue": claim.get("venue"),
                    "season": claim.get("season"),
                    "year_range": claim.get("year_range"),
                    "claimed_value": claim.get("claimed_value"),
                    "computed_value": None if computed is None else round(float(computed), 2),
                    "difference": verification.get("difference"),
                    "stats_found": bool(not rows.empty and computed is not None),
                    "stats_verified": bool(verification.get("stats_verified", False)),
                    "stat_accuracy_score": float(verification.get("stat_accuracy_score", 0.0)),
                    "stat_confidence": verification.get("stat_confidence", "low"),
                }
            )

    valid_mentions = [m for m in stat_mentions if m.get("stats_found")]
    if not valid_mentions:
        return {
            "player_mentions": mentions,
            "stat_mentions": stat_mentions,
            "stat_type": None,
            "computed_value": None,
            "claimed_value": None,
            "difference": None,
            "stats_found": False,
            "stats_verified": False,
            "stat_accuracy_score": 0.0,
            "stat_confidence": "low",
            "stat_score_adjustment": 0.0,
        }

    scores = [float(m.get("stat_accuracy_score", 0.0)) for m in valid_mentions]
    accuracy = sum(scores) / max(len(scores), 1)
    best_confidence = "high" if any(m.get("stat_confidence") == "high" for m in valid_mentions) else "medium"
    all_verified = all(bool(m.get("stats_verified", False)) for m in valid_mentions)

    has_numeric_claim = any(m.get("claimed_value") is not None for m in stat_mentions)
    weight = _writer_stat_weight(writer_type, has_numeric_claim=has_numeric_claim)
    stat_score_adjustment = round((accuracy - 0.5) * 2.0 * weight, 2)

    representative = valid_mentions[0]
    return {
        "player_mentions": mentions,
        "stat_mentions": stat_mentions,
        "stat_type": representative.get("stat_type"),
        "computed_value": representative.get("computed_value"),
        "claimed_value": representative.get("claimed_value"),
        "difference": representative.get("difference"),
        "stats_found": True,
        "stats_verified": all_verified,
        "stat_accuracy_score": round(accuracy, 4),
        "stat_confidence": best_confidence,
        "stat_score_adjustment": stat_score_adjustment,
    }
