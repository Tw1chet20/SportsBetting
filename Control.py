from main import OverUnderDataHandler, Pricer, StatisticalTester
from config import Params
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

football_file_path = Path("file_path")
df = pd.read_csv(football_file_path)

dec_bool = True
time_int = 5

params_football: Params = {
    "over_odds_cols": [
        "over_0.5",
        "over_1.5",
        "over_2.5",
        "over_3.5",
        "over_4.5"
    ],
    "under_odds_cols": [
        "under_0.5",
        "under_1.5",
        "under_2.5",
        "under_3.5",
        "under_4.5"
    ],
    "match_id_col": "match_id",
    "interval_start_col": "interval_start",
    "interval_end_col": "interval_end",
    "total_score_col": "total_goals"
}

params = params_football

dh = OverUnderDataHandler(decimal=dec_bool, time_int=time_int)
stat_tester = StatisticalTester(df=df, params=params_football)

test = stat_tester.test_poisson()

test_valid = test[test["p_value"] >= 0.05]
test_valid = test_valid[test_valid["chi2"] <= test_valid["5%_significance_level"]]
test_valid = test_valid[abs(test_valid["variance_mean_ratio"] - 1.0) <= 0.2]

print("----------")
print(f"{len(test_valid)} / {len(test)} significant tests")
print("----------")
print("\n")

valid_int_scores = pd.MultiIndex.from_frame(
    test_valid[
        [
            "interval",
            "current_score"
        ]
    ]
)
df_valid = df[
    pd.MultiIndex.from_frame(
        df[
            [
                params_football["interval_start_col"],
                params_football["total_score_col"]
            ]
        ]
    ).isin(valid_int_scores)
].reset_index()

df_ii = dh.generate_intensity(params=params, df=df_valid)
df_li = dh.generate_intensity(params=params, df=df_ii, implied=False)
U_li_surface = dh.produce_surface(df=df_li, params=params, implied=False, over=False, plot=False)

under_pricer = Pricer(surface=U_li_surface, decimal=dec_bool, time_int=time_int)

home_avg_score = 1.53
home_avg_conc = 1.37
away_avg_score = 1.00
away_avg_conc = 1.97
home_e_score = (home_avg_score + away_avg_conc) / 2
away_e_score = (away_avg_score + home_avg_conc) / 2
e_score = home_e_score + away_e_score
hist_e_score = 2.65
alpha = e_score / hist_e_score

results = []
for pair in valid_int_scores:

    for strike_score in range(6):
            
        time = pair[0]
        current_score = pair[1]

        if current_score <= strike_score:

            print("\n")
            print("-----------")
            print(
                f"(strike score,current score,current time): "
                f"({strike_score},{current_score},{time})"
            )

            price = under_pricer.over_under_price(
                strike_score=strike_score,
                strike_time=90,
                current_score=current_score,
                current_time=time,
                over=False,
                alpha=alpha
            )
            print(price)
            print("-----------")

            results.append(
                {
                    "time": time,
                    "strike_score": strike_score,
                    "current_score": current_score,
                    "price": price
                }
            )

results_full_df = pd.DataFrame(results)
results_full_df.to_csv(Path("file_path"))

time_selec = 40
results_df = results_full_df[results_full_df["time"] == time_selec]
results_df["price"] = (
    results_df["price"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)
results_df = results_df.dropna(
    subset=["price"]
)
results_df = results_df[
    results_df["price"] >= 1
]
upper_limit = results_df["price"].quantile(0.99)
results_df["price_plot"] = (
    results_df["price"]
    .clip(
        upper=upper_limit
    )
)
fig = plt.figure(
    figsize=(12, 8)
)
ax = fig.add_subplot(
    111,
    projection="3d"
)
surf = ax.plot_trisurf(
    results_df["strike_score"],
    results_df["current_score"],
    results_df["price"],
    cmap="viridis",
    linewidth=0.2,
    antialiased=True
)
ax.set_xlabel(
    "Strike Score"
)
ax.set_ylabel(
    "Current Score"
)
ax.set_zlabel(
    "Fair Decimal Odds"
)
ax.set_title(
    "Under Bet Price Surface"
)
ax.view_init(
    elev=30,
    azim=-120
)
fig.colorbar(
    surf,
    shrink=0.5,
    aspect=10,
    label="Fair Odds"
)
plt.tight_layout()
plt.show()