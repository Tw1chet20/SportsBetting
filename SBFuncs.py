import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, DefaultDict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import os
import random
from scipy.stats import lognorm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguedashteamstats,
    playergamelog,
    scoreboardv2,
)
from nba_api.stats.static import players, teams

_TEAM_DEF_CACHE: dict[str, pd.DataFrame] = {}

def nba_call_with_retry(fn, *, max_retries: int = 5, backoff_base: float = 1.6, jitter: float = 0.6):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (backoff_base ** (attempt - 1)) + random.uniform(0, jitter)
            time.sleep(sleep_s)
    raise last_err

def get_team_def_ratings(season: str) -> pd.DataFrame:
    if season in _TEAM_DEF_CACHE:
        return _TEAM_DEF_CACHE[season]

    stats = nba_call_with_retry(
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            timeout=60,
        )
    )
    df = stats.get_data_frames()[0][["TEAM_ID", "DEF_RATING"]].copy()
    df["DEF_RATING"] = df["DEF_RATING"].astype(float)
    _TEAM_DEF_CACHE[season] = df
    return df

# ----------------------------
# Betting maths
# ----------------------------

def kelly_fraction(returns: float, prob: float) -> float:
    """
    Kelly fraction for a bet with net returns `returns` (decimal_odds - 1) and win prob `prob`.
    Returns 0.0 if negative edge.
    """
    if prob <= 0 or returns <= 0:
        return 0.0
    breakeven = 1 / (returns + 1)
    return max(0.0, prob - (1 - prob) / returns) if prob > breakeven else 0.0


def should_bet(over: bool, returns: float, prob: float, stat: int):
    side = "over" if over else "under"
    min_returns_needed = -1 + 1 / prob
    return [
        kelly_fraction(returns=returns, prob=prob),
        {f"Minimum returns for {side} equal {stat} points": min_returns_needed},
    ]


def lognorm_prob(over: bool, stat: int, mean: float, var: float) -> float:
    sigma = np.sqrt(np.log(1 + (var / (mean ** 2))))
    mu = np.log(mean) - 0.5 * sigma**2
    cdf = lognorm.cdf(stat, sigma, scale=np.exp(mu))
    return float(1 - cdf) if over else float(cdf)


def expected_portfolio(bet_size: np.ndarray, probs: np.ndarray, returns: np.ndarray, portfolio: float) -> float:
    # portfolio - stake + expected payout
    return float(portfolio - bet_size.sum() + (bet_size * (1 + returns) * probs).sum())


def portfolio_outcomes(bet_size: np.ndarray, returns: np.ndarray, portfolio: float) -> Dict[str, float]:
    """
    Assumes two bets placed: [over_bet, under_bet].
    """
    pre = portfolio - bet_size.sum()
    return {
        "port_1": float(pre + bet_size[1] * (1 + returns[1])),            # only under wins
        "port_2": float(pre + (bet_size * (1 + returns)).sum()),          # both win
        "port_3": float(pre + bet_size[0] * (1 + returns[0])),            # only over wins
    }


def loss_metrics(outcomes: Dict[str, float], probs: Dict[str, float], portfolio: float, EP: float) -> Dict[str, float]:
    """
    Computes probability of loss, expected loss, max loss, expected shortfall (CVaR),
    plus a couple of convenience ratios.
    """
    vals = np.array(list(outcomes.values()), dtype=float)
    ps = np.array(list(probs.values()), dtype=float)

    losses = np.maximum(0.0, portfolio - vals)  # 0 if no loss
    loss_mask = losses > 0

    prob_loss = float(ps[loss_mask].sum())
    expected_loss = float((losses[loss_mask] * ps[loss_mask]).sum())
    max_loss = float(losses.max()) if loss_mask.any() else 0.0

    expected_shortfall = float(expected_loss / prob_loss) if prob_loss > 0 else 0.0
    cvar_to_port = float(expected_shortfall / portfolio) if portfolio > 0 else np.inf
    risk_ratio = float(expected_shortfall / EP) if EP > 0 else np.inf

    return {
        "probability_of_loss": prob_loss,
        "expected_loss": expected_loss,
        "maximum_possible_loss": max_loss,
        "expected_shortfall": expected_shortfall,
        "CVaR_relative_to_portfolio": cvar_to_port,
        "risk_ratio": risk_ratio,
    }


# ----------------------------
# Feature engineering (LR)
# ----------------------------

WINDOWS = (3, 5, 10)
BASE_FEATURES = ("IS_HOME", "Matchup_Defensive_Rating")


def _feature_cols() -> List[str]:
    cols = list(BASE_FEATURES)
    for w in WINDOWS:
        cols += [f"{x}_AVG_{w}" for x in ("PTS", "MIN", "FGA", "FTA", "FG3A")]
    return cols


FEATURE_COLS = _feature_cols()


def add_features(gameLogs: pd.DataFrame, season: str) -> pd.DataFrame:
    gl = gameLogs.copy()
    gl["IS_HOME"] = gl["MATCHUP"].astype(str).str.contains("vs").astype(int)

    for w in WINDOWS:
        gl[f"PTS_AVG_{w}"] = gl["PTS"].shift(1).rolling(w, min_periods=1).mean()
        gl[f"MIN_AVG_{w}"] = gl["MIN"].shift(1).rolling(w, min_periods=1).mean()
        gl[f"FGA_AVG_{w}"] = gl["FGA"].shift(1).rolling(w, min_periods=1).mean()
        gl[f"FTA_AVG_{w}"] = gl["FTA"].shift(1).rolling(w, min_periods=1).mean()
        gl[f"FG3A_AVG_{w}"] = gl["FG3A"].shift(1).rolling(w, min_periods=1).mean()

    # keep last row for "next game" even if partial, but ensure others are clean
    gl = pd.concat([gl.iloc[:-1].dropna(), gl.iloc[-1:]])

    # team defence ratings (advanced defence)
    df = get_team_def_ratings(season)

    opp_ids = []
    for matchup in gl["MATCHUP"].astype(str):
        abbrev = matchup.strip()[-3:]
        team = teams.find_team_by_abbreviation(abbrev)
        if team is None:
            raise ValueError(f"Could not parse opponent abbrev from MATCHUP={matchup!r} (got {abbrev!r})")
        opp_ids.append(team["id"])

    def_rtgs = df.set_index("TEAM_ID").loc[opp_ids, "DEF_RATING"].astype(float).to_list()
    gl["Matchup_Defensive_Rating"] = def_rtgs

    return gl


def add_outcome_col(over: bool, gameLogs: pd.DataFrame, stat: float) -> pd.DataFrame:
    gl = gameLogs.copy()
    col = f'{"OVER_EQUAL" if over else "UNDER_EQUAL"}_{stat}'
    gl[col] = (gl["PTS"] >= stat).astype(int) if over else (gl["PTS"] <= stat).astype(int)
    return gl


def build_and_train_LR_model(over: bool, full_gl: pd.DataFrame, stat: int):
    col = f'{"OVER_EQUAL" if over else "UNDER_EQUAL"}_{stat}'

    X = full_gl[FEATURE_COLS].iloc[:-1]
    y = full_gl[col].iloc[:-1]

    split_idx = int(len(full_gl) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    if y_train.nunique() < 2:
        return None

    clf = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
    clf.fit(X_train, y_train)
    return clf


def LR_prob(model, full_gl: pd.DataFrame) -> float:
    latest = full_gl.iloc[-1]
    next_X = pd.DataFrame([latest[FEATURE_COLS].values], columns=FEATURE_COLS)
    return float(model.predict_proba(next_X)[:, 1][0])


# ----------------------------
# Main bet pipeline
# ----------------------------

def _price_table(
    lines_to_records: Dict[int, dict],
    *,
    over: bool,
    mean: float,
    var: float,
    portfolio_size: float,
    gameLogs: pd.DataFrame,
    model_type: str,
) -> Dict[int, Dict[str, float]]:
    """
    Input:
      lines_to_records = { line: {"return": net_return, "bookmaker": "..."} }

    Output keyed by line:
      {
        line: {
          "return", "prob", "portion", "winnings",
          "bookmaker",
          "min_returns_label", "min_returns_needed"
        }
      }
    """
    out: Dict[int, Dict[str, float]] = {}

    for line, rec in lines_to_records.items():
        r = float(rec["return"])
        book = rec.get("bookmaker")

        # probability
        if model_type == "LR":
            gl = add_outcome_col(over=over, gameLogs=gameLogs, stat=line)
            model = build_and_train_LR_model(over=over, full_gl=gl, stat=line)
            if model is None:
                side = "over" if over else "under"
                print(f"LR model will not work for {side} {line} points, using lognormal probability instead")
                p = lognorm_prob(over=over, stat=line, mean=mean, var=var)
            else:
                p = LR_prob(model=model, full_gl=gl)
        else:
            p = lognorm_prob(over=over, stat=line, mean=mean, var=var)

        # bet sizing + store min-returns-needed
        kelly, min_ret_dict = should_bet(over=over, returns=r, prob=p, stat=line)
        (min_label, min_needed), = min_ret_dict.items()  # single item

        portion = float(portfolio_size * kelly)

        out[line] = {
            "return": r,
            "prob": float(p),
            "portion": portion,
            "winnings": float(portion * r),
            "bookmaker": book,
            "min_returns_label": min_label,
            "min_returns_needed": float(min_needed),
        }

    return out


def pipeline(
    over_odds: Dict[int, dict],
    under_odds: Dict[int, dict],
    mean: float,
    var: float,
    portfolio_size: float,
    gameLogs: pd.DataFrame,
    model_type: str,
) -> Optional[pd.DataFrame]:
    over_tbl = _price_table(
        over_odds, over=True, mean=mean, var=var,
        portfolio_size=portfolio_size, gameLogs=gameLogs, model_type=model_type
    )
    under_tbl = _price_table(
        under_odds, over=False, mean=mean, var=var,
        portfolio_size=portfolio_size, gameLogs=gameLogs, model_type=model_type
    )

    bets = {}
    bet_number = 1

    for over_line, o in over_tbl.items():
        if o["portion"] <= 0:
            continue
        for under_line, u in under_tbl.items():
            if u["portion"] <= 0 or not (over_line < under_line):
                continue

            portions = np.array([o["portion"], u["portion"]], dtype=float)
            probs = np.array([o["prob"], u["prob"]], dtype=float)
            returns = np.array([o["return"], u["return"]], dtype=float)

            EP = expected_portfolio(portions, probs, returns, portfolio_size)
            PO = portfolio_outcomes(portions, returns, portfolio_size)

            diff = np.array(list(PO.values())) - portfolio_size
            return_diff = float(diff.sum())

            p1 = max(0.0, 1.0 - o["prob"])
            p2 = max(0.0, o["prob"] + u["prob"] - 1.0)
            p3 = max(0.0, 1.0 - u["prob"])
            port_probs = {"port_1_prob": p1, "port_2_prob": p2, "port_3_prob": p3}

            risk = loss_metrics(PO, port_probs, portfolio_size, EP)

            bets[bet_number] = {
                "over_bet": over_line,
                "under_bet": under_line,

                "over_return": o["return"],
                "over_prob": o["prob"],
                "over_portion": o["portion"],
                "over_winnings": o["winnings"],
                "over_bookmaker": o.get("bookmaker"),
                "over_min_returns_label": o["min_returns_label"],
                "over_min_returns_needed": o["min_returns_needed"],

                "under_return": u["return"],
                "under_prob": u["prob"],
                "under_portion": u["portion"],
                "under_winnings": u["winnings"],
                "under_bookmaker": u.get("bookmaker"),
                "under_min_returns_label": u["min_returns_label"],
                "under_min_returns_needed": u["min_returns_needed"],

                **PO,
                "return_diff": return_diff,
                **port_probs,
                "EP": EP,
                **risk,
            }
            bet_number += 1

    if not bets:
        print("No bets available")
        return None

    return pd.DataFrame.from_dict(bets, orient="index")


def rand_single_bet(over: bool, stat: int, mean: float, var: float, returns: float, portfolio_size: float) -> pd.DataFrame:
    prob = lognorm_prob(over=over, stat=stat, mean=mean, var=var)
    portion = portfolio_size * should_bet(over=over, returns=returns, prob=prob, stat=stat)
    winnings = portion * returns
    EP = portfolio_size + portion * (-1 + (1 + returns) * prob)

    e_loss = portion * (1 - prob)
    CVaR = portion  # preserves your original behaviour
    CVaR_to_port = CVaR / portfolio_size
    risk_ratio = CVaR / EP if EP > 0 else np.inf

    return pd.DataFrame(
        {
            1: {
                "probability": prob,
                "amount_to_bet": portion,
                "winnings": winnings,
                "expected_portfolio": EP,
                "diff": winnings - portion,
                "expected_loss": e_loss,
                "expected_shortfall": CVaR,
                "expected_shortfall_relative_to_portfolio": CVaR_to_port,
                "risk_ratio": risk_ratio,
            }
        }
    )


# ----------------------------
# NBA + Odds API utilities
# ----------------------------

SPORT = "basketball_nba"


def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def fetch_nba_events(api_key: str, hours_ahead: int = 48) -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    resp = requests.get(url, params={"apiKey": api_key, "dateFormat": "iso"}, timeout=30)
    resp.raise_for_status()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    out = []
    for ev in resp.json():
        t = _iso_to_dt(ev["commence_time"])
        if now <= t <= cutoff:
            out.append(ev)
    return out


def fetch_player_points_for_event(
    api_key: str,
    event_id: str,
    player_name: str,
    *,
    regions: str = "uk",
    odds_format: str = "decimal",
    include_links: bool = False,
) -> Tuple[Dict[int, dict], Dict[int, dict], dict]:
    """
    Returns:
      over:  {line: {"return": net_return, "bookmaker": title}}
      under: {line: {"return": net_return, "bookmaker": title}}

    Note: preserves your original rounding behaviour on `line`.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "player_points",
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "includeLinks": "true" if include_links else "false",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    over: Dict[int, dict] = {}
    under: Dict[int, dict] = {}

    for book in data.get("bookmakers", []):
        book_title = book.get("title", book.get("key", "Unknown"))

        for market in book.get("markets", []):
            if market.get("key") != "player_points":
                continue

            for outcome in market.get("outcomes", []):
                if outcome.get("description") != player_name:
                    continue

                line = outcome.get("point")
                price = outcome.get("price")
                side = outcome.get("name")  # "Over" / "Under"

                if line is None or price is None or side not in {"Over", "Under"}:
                    continue

                net_return = float(price) - 1.0
                record = {"return": net_return, "bookmaker": book_title}

                # Preserve your original line rounding rules
                if side == "Over":
                    key = int(round(float(line) + 0.5))
                    if (key not in over) or (net_return > over[key]["return"]):
                        over[key] = record
                else:
                    key = int(round(float(line) - 0.5))
                    if (key not in under) or (net_return > under[key]["return"]):
                        under[key] = record

    meta = {
        "event_id": event_id,
        "home_team": data.get("home_team"),
        "away_team": data.get("away_team"),
        "commence_time": data.get("commence_time"),
        "regions": regions,
        "odds_format": odds_format,
    }
    return over, under, meta


def fetch_player_points_odds_best_event(
    api_key: str,
    player_name: str,
    *,
    hours_ahead: int = 48,
    regions: str = "uk",
    odds_format: str = "decimal",
    snapshot_path: Optional[str] = None,
) -> Tuple[Dict[int, dict], Dict[int, dict], dict]:
    events = fetch_nba_events(api_key, hours_ahead=hours_ahead)
    warnings: List[str] = []

    for ev in events:
        over, under, meta = fetch_player_points_for_event(
            api_key,
            ev["id"],
            player_name,
            regions=regions,
            odds_format=odds_format,
        )
        if over or under:
            out_meta = {"warnings": warnings, "picked_event": meta}

            if snapshot_path:
                snap = {
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "player": player_name,
                    "event": meta,
                    "over": over,
                    "under": under,
                }
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(snap, f, indent=2)

            return over, under, out_meta

    warnings.append(
        f"No player_points found for {player_name!r} in the next {hours_ahead} hours "
        f"for regions={regions!r}. Try regions='us' or 'us2' if UK books don’t carry these props."
    )
    return {}, {}, {"warnings": warnings, "picked_event": None}


def get_player_id(full_name: str) -> int:
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise ValueError(f"No player found for name: {full_name}")
    return matches[0]["id"]


def fetch_game_logs(
    player_id: int,
    season: str,
    *,
    timeout: int = 60,
    max_retries: int = 5,
    backoff_base: float = 1.5,
    jitter: float = 0.4,
    cache_dir: str = "nba_cache",
) -> pd.DataFrame:
    """
    Robust playergamelog fetch:
      - caches per (season, player_id) to disk
      - retries with exponential backoff + jitter
      - uses a longer timeout

    This directly addresses stats.nba.com timeouts.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"playergamelog_{season}_{player_id}.parquet")

    # Cache hit
    if os.path.exists(cache_path):
        gl = pd.read_parquet(cache_path)
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
        return gl.sort_values("GAME_DATE").reset_index(drop=True)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=timeout,
            ).get_data_frames()[0]

            gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
            gl = gl.sort_values("GAME_DATE").reset_index(drop=True)

            # Save cache
            gl.to_parquet(cache_path, index=False)
            return gl

        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise

            sleep_s = (backoff_base ** (attempt - 1)) + random.uniform(0, jitter)
            time.sleep(sleep_s)

    # Should never get here
    raise last_err


def store_active_players_with_teams(path: str = "player_team_map.csv", sleep_s: float = 0.7) -> None:
    active = players.get_active_players()
    rows = []

    for p in tqdm(active, desc="Fetching active players"):
        pid = p["id"]
        info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        team = info.loc[0, "TEAM_ABBREVIATION"]
        rows.append({"id": pid, "full_name": p["full_name"], "team": team})
        time.sleep(sleep_s)

    pd.DataFrame(rows).to_csv(path, index=False)


def upcoming_games(days_ahead: int = 2) -> Optional[pd.DataFrame]:
    rows = []

    for i in range(days_ahead):
        date = (datetime.today() + timedelta(days=i)).strftime("%m/%d/%Y")
        board = scoreboardv2.ScoreboardV2(game_date=date)
        df = board.get_data_frames()[0]

        if df.empty:
            print(f"No games available on {date}")
            continue

        rows.append(df[["GAME_DATE_EST", "GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]])

    if not rows:
        return None

    games = pd.concat(rows, ignore_index=True).dropna()

    games["HOME_TEAM"] = [teams.find_team_name_by_id(t)["abbreviation"] for t in games["HOME_TEAM_ID"]]
    games["AWAY_TEAM"] = [teams.find_team_name_by_id(t)["abbreviation"] for t in games["VISITOR_TEAM_ID"]]
    return games

# --- Odds API batching helpers ---

def _round_line_key(side: str, line: float) -> int:
    if side == "Over":
        return int(round(float(line) + 0.5))
    if side == "Under":
        return int(round(float(line) - 0.5))
    raise ValueError(f"Unexpected side: {side!r}")


def fetch_player_points_for_event_indexed(
    api_key: str,
    event_id: str,
    *,
    regions: str = "uk",
    odds_format: str = "decimal",
    bookmakers: Optional[str] = None,
    include_links: bool = False,
) -> Tuple[Dict[str, Dict[str, Dict[int, dict]]], dict]:
    """
    ONE request per event using the *event-odds* endpoint (required for player props). :contentReference[oaicite:1]{index=1}

    Returns:
      per_player[player_name]["over"|"under"][line_int] = {"return": net_return, "bookmaker": title}
      meta = {home_team, away_team, commence_time, event_id, ...}
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "player_points",
        "oddsFormat": odds_format,
        "dateFormat": "iso",
        "includeLinks": "true" if include_links else "false",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers  # overrides regions per Odds API docs :contentReference[oaicite:2]{index=2}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    meta = {
        "event_id": event_id,
        "home_team": data.get("home_team"),
        "away_team": data.get("away_team"),
        "commence_time": data.get("commence_time"),
        "regions": regions,
        "odds_format": odds_format,
        "bookmakers": bookmakers,
    }

    per_player: DefaultDict[str, Dict[str, Dict[int, dict]]] = defaultdict(lambda: {"over": {}, "under": {}})

    for book in data.get("bookmakers", []):
        book_title = book.get("title", book.get("key", "Unknown"))

        for market in book.get("markets", []):
            if market.get("key") != "player_points":
                continue

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")      # "Over"/"Under"
                line = outcome.get("point")
                price = outcome.get("price")

                if not player or side not in {"Over", "Under"} or line is None or price is None:
                    continue

                key_line = _round_line_key(side, float(line))
                net_return = float(price) - 1.0
                record = {"return": net_return, "bookmaker": book_title}

                bucket = "over" if side == "Over" else "under"
                cur = per_player[player][bucket].get(key_line)
                if (cur is None) or (net_return > cur["return"]):
                    per_player[player][bucket][key_line] = record

    return dict(per_player), meta

def _json_safe(x: Any):
    """Convert numpy/pandas types to JSON-safe python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return x

def save_successful_bet(
    *,
    out_dir: str,
    player_name: str,
    matchup: str,
    game_date: str,
    event_meta: dict,
    odds_snapshot: dict,
    bets_df: pd.DataFrame,
    top_n: int = 1,
) -> str:
    """
    Writes:
      1) JSON snapshot of odds + bet results
      2) Appends a compact summary row to a JSONL "ledger"

    Returns the path to the saved JSON snapshot.
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Choose best rows to store (default: top 1). You can change ranking logic.
    # If you want a different criterion, swap the sort key.
    df = bets_df.copy()

    # Example ranking: highest EP, then lowest risk_ratio
    if "EP" in df.columns and "risk_ratio" in df.columns:
        df = df.sort_values(["EP", "risk_ratio"], ascending=[False, True])
    else:
        df = df.sort_values(df.columns[0])  # fallback deterministic

    top_rows = df.head(top_n)

    payload = {
        "saved_at_utc": ts,
        "player_name": player_name,
        "matchup": matchup,
        "game_date": str(game_date),
        "event": event_meta,
        "odds_snapshot": odds_snapshot,   # contains over/under with bookmaker + returns
        "bets_top": json.loads(top_rows.to_json(orient="records", default_handler=_json_safe)),
        "bets_all": json.loads(df.to_json(orient="records", default_handler=_json_safe)),
    }

    # File names safe-ish
    safe_player = "".join(c for c in player_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    safe_match = "".join(c for c in matchup if c.isalnum() or c in (" ", "_", "-", "@", ".")).strip().replace(" ", "_")

    snapshot_path = os.path.join(out_dir, f"{ts}__{safe_player}__{safe_match}.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)

    # Append a compact ledger entry (JSONL)
    ledger_path = os.path.join(out_dir, "successful_bets.jsonl")
    best = payload["bets_top"][0] if payload["bets_top"] else {}

    ledger_row = {
        "saved_at_utc": ts,
        "player_name": player_name,
        "matchup": matchup,
        "game_date": str(game_date),
        "event_id": event_meta.get("event_id"),
        "home_team": event_meta.get("home_team"),
        "away_team": event_meta.get("away_team"),
        # key bet fields (these include min returns if you added them to pipeline output)
        "over_bet": best.get("over_bet"),
        "under_bet": best.get("under_bet"),
        "over_return": best.get("over_return"),
        "under_return": best.get("under_return"),
        "over_bookmaker": best.get("over_bookmaker"),
        "under_bookmaker": best.get("under_bookmaker"),
        "over_min_returns_needed": best.get("over_min_returns_needed"),
        "under_min_returns_needed": best.get("under_min_returns_needed"),
        "EP": best.get("EP"),
        "risk_ratio": best.get("risk_ratio"),
        "snapshot_file": os.path.basename(snapshot_path),
    }

    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_row, default=_json_safe) + "\n")

    return snapshot_path