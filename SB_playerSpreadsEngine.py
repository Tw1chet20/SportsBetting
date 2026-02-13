import warnings
warnings.simplefilter("ignore", FutureWarning)

import pandas as pd
from tqdm import tqdm
import time
import SBFuncs as SPF
import nba_api.library.http as nba_http

nba_http.TIMEOUT = 60

API_KEY = "1e8733ed806ebdfaeb16ea967d7e4b0e"
portfolio_size = 70.91
season = "2025-26"

# Optional: limit bookmakers to reduce payload/latency (still 1 request per game)
BOOKMAKERS = None  # e.g. "bet365,williamhill"

games = SPF.upcoming_games(days_ahead=2)
if games is None or games.empty:
    raise SystemExit("No upcoming games found.")

active_players = pd.read_csv(
    "/Users/tristanwinter/Documents/Documents - Tristan’s MacBook Air/VSCode/LearningPython/.venv/sportsBetting/player_team_map.csv"
)

# Odds API events once
events = SPF.fetch_nba_events(API_KEY, hours_ahead=48)

# Build abbrev -> full name map for matching
abbrev_to_full = {}
for abbrev in pd.unique(pd.concat([games["HOME_TEAM"], games["AWAY_TEAM"]])):
    t = SPF.teams.find_team_by_abbreviation(abbrev)
    if t:
        abbrev_to_full[abbrev] = t["full_name"]

def match_event_id(home_abbrev: str, away_abbrev: str) -> str | None:
    home_full = abbrev_to_full.get(home_abbrev)
    away_full = abbrev_to_full.get(away_abbrev)
    if not home_full or not away_full:
        return None

    for ev in events:
        # Odds API uses full names for home_team/away_team
        if ev.get("home_team") == home_full and ev.get("away_team") == away_full:
            return ev["id"]
    return None

for _, g in games.iterrows():
    game_date = g["GAME_DATE_EST"]
    home = g["HOME_TEAM"]
    away = g["AWAY_TEAM"]

    event_id = match_event_id(home, away)
    if not event_id:
        print(f"No Odds API event matched for {away} @ {home}")
        continue

    # ONE Odds API call per game
    per_player_props, meta = SPF.fetch_player_points_for_event_indexed(
        API_KEY,
        event_id,
        regions="us",
        odds_format="decimal",
        bookmakers=BOOKMAKERS,
    )

    if not per_player_props:
        print(f"No player_points returned for {away} @ {home}")
        continue

    # Only consider players on these teams
    game_players = pd.concat(
        [
            active_players.loc[active_players["team"] == home, ["full_name"]],
            active_players.loc[active_players["team"] == away, ["full_name"]],
        ],
        ignore_index=True,
    )["full_name"].tolist()

    for player_name in tqdm(game_players, desc=f"Finding bets {away} @ {home}", leave=False):
        props = per_player_props.get(player_name)
        if not props:
            continue

        over_raw = props["over"]
        under_raw = props["under"]
        if not over_raw or not under_raw:
            continue

        over_odds = over_raw
        under_odds = under_raw

        pid = SPF.get_player_id(player_name)
        df_api = SPF.fetch_game_logs(pid, season)
        time.sleep(0.15)

        # Build next-game row for features
        next_matchup = f"{home} vs. {away}"
        next_game_log = [
            df_api["SEASON_ID"].iloc[-1],
            df_api["Player_ID"].iloc[-1],
            None,
            game_date,
            next_matchup,
        ]
        next_game_log += [None] * (len(df_api.columns) - len(next_game_log))
        next_game = pd.DataFrame([next_game_log], columns=df_api.columns)

        df_api_full = SPF.add_features(gameLogs=pd.concat([df_api, next_game], ignore_index=True), season=season)

        pts = pd.to_numeric(df_api["PTS"], errors="coerce").dropna()
        if pts.empty:
            continue
        mean, var = float(pts.mean()), float(pts.var())

        bets_df = SPF.pipeline(
            over_odds=over_odds,
            under_odds=under_odds,
            mean=mean,
            var=var,
            portfolio_size=portfolio_size,
            gameLogs=df_api_full,
            model_type="lognorm",
        )

        odds_snapshot = {
    "over": over_raw,   # {line: {"return": ..., "bookmaker": ...}}
    "under": under_raw,
}

if bets_df is not None and not bets_df.empty:
    snapshot_file = SPF.save_successful_bet(
        out_dir="bet_outputs",
        player_name=player_name,
        matchup=f"{away} @ {home}",
        game_date=game_date,
        event_meta=meta,                 
        odds_snapshot=odds_snapshot,
        bets_df=bets_df,
        top_n=1,
    )
    print(f"Saved successful bet snapshot: {snapshot_file}")