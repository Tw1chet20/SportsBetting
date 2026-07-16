import pandas as pd
import numpy as np
from config import Params
from scipy.optimize import brentq
from scipy.special import gammaincc, gamma
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import exp, factorial, log
from typing import Dict
from tqdm import tqdm
from scipy.stats import poisson, chisquare

class OverUnderDataHandler():

    def __init__(self, decimal: bool, time_int: int):
        self.decimal = decimal
        self.time_int = time_int

    def dupire_imp(
            self,
            K: int,
            F: float,
            over: bool
            ) -> float:

        if F == 1.0:
            return 0.0
        elif np.isnan(F):
            return np.nan

        if over:
            f = lambda lam: 1 - (gamma(K+1) * gammaincc(K+1, lam) / factorial(K)) + (exp(-lam) * (lam**K) / factorial(K)) - F
            return brentq(f, 0.0, 1000.0)
        else:
            f = lambda lam: (gamma(K+1) * gammaincc(K+1, lam) / factorial(K)) - F
            val = brentq(f, 0.0, 1000.0)
            if val > 0.0:
                return val
            else:
                return 0.0
    
    def dupire_loc(
            self,
            K: int,
            T: int,
            lam: float,
            lam_prev: float,
            gradients: Dict
            ) -> float:
        
        if lam == 0.0 or lam_prev == 0.0:
            return 0.0
        elif np.isnan(lam) or np.isnan(lam_prev):
            return np.nan
        
        num = exp(-lam) * (lam**K) * gradients[(T,K)][0]
        denom = (gammaincc(K+1, lam) * gamma(K+1)) - (K * gammaincc(K, lam_prev) * gamma(K))

        if denom == 0.0:
            return np.inf
            
        val = - num / denom

        if val > 0.0:
            return val
        else:
            return 0.0
        
    def gradients(self, surface: pd.DataFrame) -> Dict:
        
        Z = surface.to_numpy(dtype=float)
        times = surface.columns.to_numpy()
        strikes = surface.index.to_numpy()

        dI_dk, dI_dt = np.gradient(Z)

        records = {}

        for i, strike in enumerate(strikes):
            for j, time in enumerate(times):
                records[(int(time),int(strike))] = (dI_dt[i, j], dI_dk[i, j])

        return records

    def generate_intensity(
            self,
            params: Params,
            df: pd.DataFrame,
            implied: bool = True
            ) -> pd.DataFrame:
        
        df_copy = df.copy()
        
        for index_over in range(len(params["over_odds_cols"])):

            k = ''.join(c for c in params["over_odds_cols"][index_over] if c.isdigit())

            K = int(float(k[:-1] + '.' + k[-1]) + 0.5)

            if implied:
                if index_over == 0:
                    df_copy[f"over_equal_{K}_implied_intensity"] = df_copy[params["over_odds_cols"][index_over]].apply(lambda x: 1 + log(1/(float(x) + (1.0 - int(self.decimal)))))
                else:
                    df_copy[f"over_equal_{K}_implied_intensity"] = df_copy[params["over_odds_cols"][index_over]].apply(lambda x: self.dupire_imp(K, 1/(float(x) + (1.0 - int(self.decimal))), over=True))
            else:
                imp_surface = self.produce_surface(
                    df=df_copy,
                    params=params,
                    over=True,
                    plot=False,
                    implied=True
                    )
                imp_gradients = self.gradients(imp_surface)
                
                df_copy[f"over_equal_{K}_local_intensity"] = [np.nan]*len(df_copy)
                for i in range(0, len(df_copy)):
                    if f"over_equal_{K-1}_implied_intensity" in df_copy.columns:
                        lam = df_copy.loc[i, f"over_equal_{K}_implied_intensity"]
                        lam_prev = df_copy.loc[i, f"over_equal_{K-1}_implied_intensity"]
                        T = int(df_copy.loc[i, params["interval_start_col"]])
                        df_copy.loc[i, f"over_equal_{K}_local_intensity"] = self.dupire_loc(K=K, T=T, lam=lam, lam_prev=lam_prev, gradients=imp_gradients)
                    else:
                        T = int(df_copy.loc[i, params["interval_start_col"]])
                        df_copy.loc[i, f"over_equal_{K}_local_intensity"] = -imp_gradients[(T,K)][0]

        for index_under in range(len(params["under_odds_cols"])):

            k = ''.join(c for c in params["under_odds_cols"][index_under] if c.isdigit())

            K = int(float(k[:-1] + '.' + k[-1]) - 0.5)

            if implied:
                if index_under == 0:
                    df_copy[f"under_equal_{K}_implied_intensity"] = df_copy[params["under_odds_cols"][index_under]].apply(lambda x: -log(1/(float(x) + (1.0 - int(self.decimal)))) if -log(1/(float(x) + (1.0 - int(self.decimal)))) > 0.0 else 0.0)
                else:
                    df_copy[f"under_equal_{K}_implied_intensity"] = df_copy[params["under_odds_cols"][index_under]].apply(lambda x: self.dupire_imp(K, 1/(float(x) + (1.0 - int(self.decimal))), over=False))
            else:
                imp_surface = self.produce_surface(
                    df=df_copy,
                    params=params,
                    over=False,
                    plot=False,
                    implied=True
                    )
                imp_gradients = self.gradients(imp_surface)
                
                df_copy[f"under_equal_{K}_local_intensity"] = [np.nan]*len(df_copy)
                for i in range(0, len(df_copy)):
                    if f"under_equal_{K-1}_implied_intensity" in df_copy.columns:
                        lam = df_copy.loc[i, f"under_equal_{K}_implied_intensity"]
                        lam_prev = df_copy.loc[i, f"under_equal_{K-1}_implied_intensity"]
                        T = int(df_copy.loc[i, params["interval_start_col"]])
                        df_copy.loc[i, f"under_equal_{K}_local_intensity"] = self.dupire_loc(K=K, T=T, lam=lam, lam_prev=lam_prev, gradients=imp_gradients)
                    else:
                        T = int(df_copy.loc[i, params["interval_start_col"]])
                        df_copy.loc[i, f"under_equal_{K}_local_intensity"] = -imp_gradients[(T,K)][0] if -imp_gradients[(T,K)][0] > 0.0 else 0.0

        return df_copy
    
    def produce_surface(
            self,
            df: pd.DataFrame,
            params: Params,
            over: bool,
            plot: bool = True,
            implied: bool = True,
            log_plot: bool = False,
            both: bool = False
            ) -> pd.DataFrame:
        
        df_copy = df.copy()

        intensity_type = "implied" if implied else "local"
        over_type = "over_equal_" if over else "under_equal_"
        over_type = "over & under" if both else over_type
        log_type = "log " if log_plot else ""

        strike_cols = [
            col for col in df_copy.columns
            if over_type in col
        ] if not both else [
            col for col in df_copy.columns
            if ("over_equal_" in col) or ("under_equal_" in col)
        ]
        strike_cols = [
            col for col in strike_cols
            if f"_{intensity_type}_intensity" in col
        ]
        
        long_df = df.melt(
            id_vars=[
                params["match_id_col"],
                params["interval_start_col"],
                params["interval_end_col"]
            ],
            value_vars=strike_cols,
            var_name="strike",
            value_name=f"{intensity_type}_intensity"
        )

        long_df["strike"] = (
            long_df["strike"]
            .apply(lambda col: int(''.join(c for c in col if c.isdigit())))
        )

        surface_data = (
            long_df
            .groupby(
                [
                    params["interval_start_col"],
                    "strike"
                ],
                as_index=False
            )
            [f"{intensity_type}_intensity"]
            .mean()
        )

        surface = surface_data.pivot(
            index="strike",
            columns=params["interval_start_col"],
            values=f"{intensity_type}_intensity"
        )

        surface = surface.apply(
            pd.to_numeric,
            errors="coerce"
        )

        surface_log = np.log(surface)

        X_values = surface.columns.values
        Y_values = surface.index.values

        X, Y = np.meshgrid(
            X_values,
            Y_values
        )

        Z = surface_log.values.astype(float) if log_plot else surface.values.astype(float)

        Z[~np.isfinite(Z)] = np.nan

        if plot:

            fig = plt.figure(
                figsize=(12, 8)
            )

            ax = fig.add_subplot(
                111,
                projection="3d"
            )

            ax.plot_surface(
                X,
                Y,
                Z,
                cmap="viridis",
                linewidth=0,
                antialiased=True
            )

            ax.set_xlabel(
                "Match interval (minutes)"
            )

            ax.set_ylabel(
                "Score strike"
            )

            ax.set_zlabel(
                f"{log_type}average {over_type} {intensity_type} intensity"
            )

            ax.set_title(
                f"{log_type}average in-play score {over_type} {intensity_type} intensity surface"
            )

            ax.view_init(
                elev=30,
                azim=-120
            )

            plt.tight_layout()

            plt.show()

        return surface

class StatisticalTester():

    def __init__(
            self,
            df: pd.DataFrame,
            params: Params
            ):
        self.df = df
        self.params = params

    def test_poisson(self, plot: bool = False):

        sig_chi2 = {
            5:
            {
                1:3.84,
                2:5.99,
                3:7.81,
                4:9.49,
                5:11.07,
                6:12.59,
                7:14.07,
                8:15.51,
                9:16.92,
                10:18.31
            },
            1:
            {
                1:6.64,
                2:9.21,
                3:11.35,
                4:13.28,
                5:15.09,
                6:16.81,
                7:18.48,
                8:20.09,
                9:21.67,
                10:23.21
            }
        }

        df_copy = self.df.copy()

        final_score = (
            df_copy.sort_values(
                [self.params["match_id_col"], self.params["interval_start_col"]]
            )
            .groupby(self.params["match_id_col"])[self.params["total_score_col"]]
            .last()
        )

        df_copy["final_score"] = (
            df_copy[self.params["match_id_col"]]
            .map(final_score)
        )

        df_copy["remaining_scores"] = (
            df_copy["final_score"] -
            df_copy[self.params["total_score_col"]]
        )

        results = []

        for (interval, current_score), group in df_copy.groupby(
            [
                self.params["interval_start_col"],
                self.params["total_score_col"]
            ]
        ):

            observed = group["remaining_scores"]

            if len(observed) < 100:
                continue

            deg_free = max(observed) - 2

            lam = observed.mean()

            max_goals = int(observed.max())

            k = np.arange(max_goals + 1)

            observed_counts = (
                observed
                .value_counts()
                .reindex(k, fill_value=0)
                .sort_index()
            )

            expected_probs = poisson.pmf(
                k,
                lam
            )

            expected_probs[-1] += (
                1 -
                expected_probs.sum()
            )

            expected_counts = (
                expected_probs *
                len(observed)
            )

            chi2, p_value = chisquare(
                observed_counts,
                expected_counts
            )

            results_dict = {
                "interval": interval,
                "current_score": current_score,
                "lambda": lam,
                "sample_size": len(observed),
                "mean": observed.mean(),
                "variance": observed.var(),
                "variance_mean_ratio": observed.var() / observed.mean() if observed.mean() != 0 else np.nan,
                "chi2": chi2,
                "1%_significance_level": sig_chi2[1][deg_free] if deg_free > 0 else 0.0,
                "5%_significance_level": sig_chi2[5][deg_free] if deg_free > 0 else 0.0,
                "p_value": p_value
            }

            results.append(results_dict)

            if plot:

                print("\n")
                print(results_dict)

                plt.figure(figsize=(8,5))

                plt.bar(
                    k,
                    observed_counts,
                    width=0.4,
                    label="Observed"
                )

                plt.plot(
                    k,
                    expected_counts,
                    "o-",
                    linewidth=2,
                    label="Poisson"
                )

                plt.xlabel("Remaining score")
                plt.ylabel("Count")

                plt.title(f"Minute {interval}, Current Score {current_score}\nλ={lam:.2f}")

                plt.legend()

                plt.show()

        results_df = pd.DataFrame(results)

        return results_df


class Pricer():

    def __init__(
            self,
            surface: pd.DataFrame,
            decimal: bool,
            time_int: int
            ):
        self.surface = surface
        self.decimal = decimal
        self.time_int = time_int

    def interpolate_surface(self, strike: int, time: int) -> float:

        times = self.surface.columns.values.astype(float)
        strikes = self.surface.index.values.astype(float)
        Z = self.surface.values

        intensity_surface = RegularGridInterpolator(
            (
                strikes,
                times
            ),
            Z,
            method="linear",
            bounds_error=False,
            fill_value=None
        )

        return intensity_surface([strike,time])[0]

    def over_under_price(
            self,
            strike_score: int,
            strike_time: int,
            current_score: int,
            current_time: int,
            over: bool,
            alpha: float
            ) -> float:

        N = 50000

        total_goals = []

        for sim in tqdm(range(N)):

            time = current_time
            goals = current_score

            while time < strike_time:

                lam = self.time_int * alpha * self.interpolate_surface(goals,time)

                goal_potential = 5

                while goal_potential != 0:

                    prob = exp(-lam) * (lam ** goal_potential) / factorial(goal_potential)

                    if np.random.rand() <= prob:
                        goals += goal_potential
                        break

                    goal_potential -= 1

                time += self.time_int

            if over:
                total_goals.append(1) if goals >= strike_score else total_goals.append(0)
            else:
                total_goals.append(1) if goals <= strike_score else total_goals.append(0)

        wins = sum(total_goals) / N

        if wins == 0.0:
            return np.inf
        else:
            return 1/(wins + (1.0 - int(self.decimal)))