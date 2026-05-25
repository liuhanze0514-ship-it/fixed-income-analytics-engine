"""
Credit Spread Anomaly Detection
===============================
Uses sklearn's IsolationForest to detect outliers in bond credit spreads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


class SpreadAnomalyDetector:
    """
    Detect anomalous credit spreads in bond pricing using Isolation Forest.

    The Isolation Forest algorithm identifies observations that are few and
    different — i.e. likely anomalies — by randomly partitioning the feature
    space and measuring the path length required to isolate each point.

    Parameters
    ----------
    contamination : float or 'auto', default 'auto'
        Expected proportion of outliers.  'auto' uses the algorithm default
        (usually ~0.1).
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(self, contamination="auto", random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.data_ = None
        self.labels_ = None
        self.scores_ = None

    # ------------------------------------------------------------------
    # Prepare dummy bond data
    # ------------------------------------------------------------------
    @staticmethod
    def prepare_data(n_bonds=100, n_anomalies=5, seed=42):
        """
        Generate a dummy bond dataset with known anomalies injected.

        The dataset simulates investment-grade and high-yield corporate bonds
        with features:
            - spread          : credit spread (bps over risk-free)
            - duration        : modified duration (years)
            - rating_num      : numeric rating (1 = AAA, ..., 21 = C)
            - issue_size      : issuance amount ($bn)
            - time_to_maturity: years until maturity

        Anomalies are injected by perturbing a small number of bonds to have
        spreads far outside their rating-implied range.

        Parameters
        ----------
        n_bonds : int
            Total number of bonds (including anomalies).
        n_anomalies : int
            Number of anomalous bonds to inject.
        seed : int
            Random seed.

        Returns
        -------
        pd.DataFrame
            Columns: spread, duration, rating_num, issue_size,
                     time_to_maturity, is_anomaly
        """
        rng = np.random.default_rng(seed)

        # Generate base parameters for investment-grade and high-yield
        n_ig = int(n_bonds * 0.7)          # ~70% IG
        n_hy = n_bonds - n_anomalies - n_ig  # remaining are HY

        # -- Investment-grade bonds (ratings 1-10) --
        ig_rating = rng.integers(1, 11, size=n_ig)
        ig_spread = 5.0 + ig_rating * 8.0 + rng.normal(0, 5, n_ig)      # spreads 10–120 bps
        ig_duration = rng.uniform(1.5, 12.0, n_ig)
        ig_issue_size = rng.uniform(0.3, 5.0, n_ig)
        ig_ttm = rng.uniform(1.0, 30.0, n_ig)

        # -- High-yield bonds (ratings 11-21) --
        hy_rating = rng.integers(11, 22, size=n_hy)
        hy_spread = 100.0 + (hy_rating - 10) * 25.0 + rng.normal(0, 20, n_hy)  # 120–450 bps
        hy_duration = rng.uniform(1.0, 8.0, n_hy)
        hy_issue_size = rng.uniform(0.1, 2.5, n_hy)
        hy_ttm = rng.uniform(1.0, 20.0, n_hy)

        # Concatenate normal bonds
        rating = np.concatenate([ig_rating, hy_rating])
        spread = np.concatenate([ig_spread, hy_spread])
        duration = np.concatenate([ig_duration, hy_duration])
        issue_size = np.concatenate([ig_issue_size, hy_issue_size])
        ttm = np.concatenate([ig_ttm, hy_ttm])
        is_anomaly = np.zeros(len(rating), dtype=int)

        # -- Inject anomalies --
        # Artificially inflate spread for a random subset of bonds
        anomaly_idx = rng.choice(len(rating), size=n_anomalies, replace=False)
        spread[anomaly_idx] = rng.uniform(600, 1200, n_anomalies)
        # Also make durations somewhat unusual
        duration[anomaly_idx] = rng.uniform(20, 40, n_anomalies)
        is_anomaly[anomaly_idx] = 1

        df = pd.DataFrame({
            "spread": spread,
            "duration": duration,
            "rating_num": rating,
            "issue_size": issue_size,
            "time_to_maturity": ttm,
            "is_anomaly": is_anomaly,
        })

        return df

    # ------------------------------------------------------------------
    # Fit the Isolation Forest
    # ------------------------------------------------------------------
    def fit(self, df, feature_cols=None):
        """
        Fit IsolationForest on the selected feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Bond data.
        feature_cols : list of str, optional
            Columns to use as features.  Defaults to:
            ['spread', 'duration', 'rating_num', 'issue_size',
             'time_to_maturity'].

        Returns
        -------
        self : SpreadAnomalyDetector
        """
        if feature_cols is None:
            feature_cols = [
                "spread", "duration", "rating_num",
                "issue_size", "time_to_maturity",
            ]

        self.data_ = df.copy()
        X = self.data_[feature_cols].values

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # Predict:  1 = inlier, -1 = anomaly
        raw_labels = self.model.predict(X)
        self.labels_ = np.where(raw_labels == -1, 1, 0)

        # Decision scores (negative = more anomalous)
        self.scores_ = self.model.decision_function(X)

        self.data_["anomaly_pred"] = self.labels_
        self.data_["anomaly_score"] = self.scores_

        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self):
        """Print detection summary."""
        if self.model is None:
            raise RuntimeError("Call fit() before summary().")

        total = len(self.labels_)
        n_anomalies = self.labels_.sum()
        pct = n_anomalies / total * 100

        print("=== Spread Anomaly Detection Summary ===")
        print(f"  Total bonds        : {total}")
        print(f"  Anomalies detected : {n_anomalies} ({pct:.1f} %)")
        print(f"  Mean anomaly score : {self.scores_[self.labels_ == 1].mean():.4f}")
        print(f"  Mean inlier score  : {self.scores_[self.labels_ == 0].mean():.4f}")

        if "is_anomaly" in self.data_.columns:
            tp = ((self.labels_ == 1) & (self.data_["is_anomaly"] == 1)).sum()
            fp = ((self.labels_ == 1) & (self.data_["is_anomaly"] == 0)).sum()
            fn = ((self.labels_ == 0) & (self.data_["is_anomaly"] == 1)).sum()
            print(f"  True anomalies     : {self.data_['is_anomaly'].sum()}")
            print(f"  True positives     : {tp}")
            print(f"  False positives    : {fp}")
            print(f"  False negatives    : {fn}")

        return self.data_

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(self, save_path=None, show=False):
        """
        Scatter plot of credit spread vs duration, colour-coded by prediction.

        Parameters
        ----------
        save_path : str or None
            If given, saves the figure to this path.
        show : bool
            If True, displays the plot interactively.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before plot().")

        df = self.data_
        normal = df[df["anomaly_pred"] == 0]
        anomaly = df[df["anomaly_pred"] == 1]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            normal["duration"], normal["spread"],
            c="steelblue", s=50, alpha=0.6, label="Normal",
        )
        ax.scatter(
            anomaly["duration"], anomaly["spread"],
            c="crimson", s=90, alpha=0.9, marker="X",
            edgecolors="black", linewidths=0.5, label="Anomaly",
        )

        ax.set_xlabel("Modified Duration (years)")
        ax.set_ylabel("Credit Spread (bps)")
        ax.set_title("Isolation Forest – Credit Spread Anomaly Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Plot Results
    # ------------------------------------------------------------------
    def plot_results(self, save_path=None, show=False):
        """
        Dashboard-style plot showing normal bonds vs flagged anomalies.

        Creates two side-by-side panels:
          Left  – Spread vs Duration scatter, coloured by prediction.
          Right – Anomaly score histogram with detection threshold.

        Parameters
        ----------
        save_path : str or None
            If given, saves the figure to this path.
        show : bool
            If True, displays the plot interactively.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before plot_results().")

        df = self.data_
        normal = df[df["anomaly_pred"] == 0]
        anomaly = df[df["anomaly_pred"] == 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Left panel: Spread vs Duration ---
        ax1.scatter(
            normal["duration"], normal["spread"],
            c="#2ecc71", s=45, alpha=0.65, label="Normal",
        )
        ax1.scatter(
            anomaly["duration"], anomaly["spread"],
            c="#e74c3c", s=80, alpha=0.9, marker="X",
            edgecolors="black", linewidths=0.5, label="Flagged Anomaly",
        )
        ax1.set_xlabel("Modified Duration (years)", fontsize=11)
        ax1.set_ylabel("Credit Spread (bps)", fontsize=11)
        ax1.set_title("Spread vs Duration", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.25)

        # --- Right panel: Anomaly score histogram ---
        scores = self.scores_
        threshold = np.percentile(scores, self.contamination * 100) if isinstance(self.contamination, (int, float)) else np.percentile(scores, 10)

        ax2.hist(
            scores[df["anomaly_pred"] == 0],
            bins=25, color="#2ecc71", alpha=0.7, label="Normal",
        )
        ax2.hist(
            scores[df["anomaly_pred"] == 1],
            bins=15, color="#e74c3c", alpha=0.85, label="Flagged Anomaly",
        )
        ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold ≈ {threshold:.3f}")
        ax2.set_xlabel("Anomaly Score (lower = more anomalous)", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Anomaly Score Distribution", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.25)

        fig.suptitle(
            "Credit Spread Anomaly Detection — Isolation Forest Results",
            fontsize=15, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ======================================================================
# Demonstration
# ======================================================================
if __name__ == "__main__":
    # 1. Prepare dummy bond data with known anomalies
    detector = SpreadAnomalyDetector(contamination=0.05, random_state=42)
    df = detector.prepare_data(n_bonds=100, n_anomalies=5, seed=42)

    print("Sample of prepared data (first 8 rows):")
    print(df.head(8).to_string(index=False))
    print()

    # 2. Fit Isolation Forest
    detector.fit(df)

    # 3. Summary
    result_df = detector.summary()

    print("\nAnomalous bonds detected:")
    print(result_df[result_df["anomaly_pred"] == 1].to_string(index=False))

    # 4. Plot results and save
    detector.plot_results(save_path="spread_anomalies.png", show=False)
    print("\nPlot saved to spread_anomalies.png")
