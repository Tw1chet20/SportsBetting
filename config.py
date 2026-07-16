from typing import TypedDict, List

class Params(TypedDict):
    over_odds_cols: List[str]
    under_odds_cols: List[str]
    match_id_col: str
    interval_start_col: str
    interval_end_col: str
    total_score_col: str