import numpy as np
import pandas as pd
from scipy.stats import poisson
import random
from datetime import datetime, timedelta
from pathlib import Path


# -----------------------------
# Configuration
# -----------------------------

OUTPUT_FILE = Path("/Users/tristanwinter/SportsBetting/csv_files/synthetic_epl_live_odds_dataset.csv")

INTERVALS = list(range(0, 90, 5))

TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford",
    "Brighton", "Chelsea", "Crystal Palace", "Everton",
    "Fulham", "Leeds", "Liverpool", "Man City",
    "Man United", "Newcastle", "Nottingham Forest",
    "Sunderland", "Tottenham", "West Ham",
    "Wolves", "Burnley"
]

SEASONS = [
    "2021_22",
    "2022_23",
    "2023_24",
    "2024_25",
    "2025_26"
]

# Team attacking strength multipliers
TEAM_STRENGTH = {
    team: np.random.uniform(0.85, 1.25)
    for team in TEAMS
}


# -----------------------------
# Utility functions
# -----------------------------

def bookmaker_odds(probability, margin=0.05):

    probability = np.clip(probability, 0.01, 0.999999)

    fair_odds = 1 / probability

    odds = fair_odds * (1 - margin)

    # Decimal odds can never be below 1.00
    return round(max(1.00, odds), 2)


def goal_probability_over(line, expected_goals):
    """
    Probability of finishing above goal line.
    """

    goals_needed = int(np.floor(line)) + 1

    probability = (
        1 -
        poisson.cdf(goals_needed - 1, expected_goals)
    )

    return probability


def generate_market_odds(total_goals,
                          minutes_remaining,
                          home_strength,
                          away_strength):

    """
    Generate live totals markets.
    """

    # Remaining expected goals
    attacking_factor = (
        home_strength +
        away_strength
    ) / 2


    remaining_xg = (
        minutes_remaining / 90
    ) * 2.7 * attacking_factor


    future_total_expectation = (
        total_goals +
        remaining_xg
    )


    markets = {}

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:

        required_goals = int(line) + 1

        # Over already settled as won
        if total_goals >= required_goals:
            markets[f"over_{line}"] = 1.00
            markets[f"under_{line}"] = 1.00
            continue


        # Under still possible, calculate live odds
        over_probability = goal_probability_over(
            line,
            future_total_expectation
        )

        under_probability = 1 - over_probability


        markets[f"over_{line}"] = bookmaker_odds(
            over_probability
        )

        markets[f"under_{line}"] = bookmaker_odds(
            under_probability
        )

    return markets



# -----------------------------
# Generate one match
# -----------------------------

def generate_match(match_id,
                   home,
                   away,
                   date,
                   season):

    rows = []

    home_strength = TEAM_STRENGTH[home]
    away_strength = TEAM_STRENGTH[away]


    # Full match goal expectation

    home_xg = (
        1.35 *
        home_strength
    )

    away_xg = (
        1.15 *
        away_strength
    )


    home_goals = 0
    away_goals = 0


    for minute in INTERVALS:


        # Probability of goals in next 5 minutes

        home_interval_lambda = (
            home_xg / 18
        )

        away_interval_lambda = (
            away_xg / 18
        )


        h_goals = np.random.poisson(
            home_interval_lambda
        )

        a_goals = np.random.poisson(
            away_interval_lambda
        )


        home_goals += h_goals
        away_goals += a_goals


        total_goals = (
            home_goals +
            away_goals
        )


        odds = generate_market_odds(
            total_goals,
            90 - minute,
            home_strength,
            away_strength
        )


        row = {

            "season": season,
            "match_id": match_id,
            "date": date,

            "interval_start": minute,
            "interval_end": minute + 5,

            "home_team": home,
            "away_team": away,

            "home_goals_interval": h_goals,
            "away_goals_interval": a_goals,

            "home_goals_total": home_goals,
            "away_goals_total": away_goals,

            "total_goals": total_goals,

            "score":
                f"{home_goals}-{away_goals}",

            **odds
        }


        rows.append(row)


    return rows



# -----------------------------
# Generate Premier League season
# -----------------------------

def generate_season(season, match_id_start):

    fixtures = []

    match_id = match_id_start

    for home in TEAMS:

        for away in TEAMS:

            if home != away:

                fixtures.append(
                    (
                        match_id,
                        home,
                        away
                    )
                )

                match_id += 1

    random.shuffle(fixtures)

    fixtures = fixtures[:380]

    all_rows = []

    start_date = datetime(int(season.split("_")[0]),8,15)


    for match_id, home, away in fixtures:

        match_date = (
            start_date +
            timedelta(days=random.randint(0,280))
        )


        match_rows = generate_match(
            match_id,
            home,
            away,
            match_date.date(),
            season
        )

        all_rows.extend(match_rows)


        print(
            f"Generated match {match_id}/{match_id_start+379}"
        )


    return pd.DataFrame(all_rows)



# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    all_seasons = []

    match_id = 1

    for season in SEASONS:

        print("----------")
        print(f"Generating {season}")

        season_df = generate_season(season, match_id_start=match_id)

        all_seasons.append(season_df)

        match_id += 380

    df = pd.concat(
        all_seasons,
        ignore_index=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(df.shape)