"""
Nelson-Siegel Yield Curve Fitting Engine
========================================
Fits the Nelson-Siegel model to market yield data and plots the resulting curve.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class YieldCurveFitter:
    """
    Nelson-Siegel yield curve fitter.

    The Nelson-Siegel formula for the instantaneous forward rate is:

        f(t) = Beta0 + Beta1 * exp(-t / Tau) + Beta2 * (t / Tau) * exp(-t / Tau)

    The spot (zero-coupon) rate for maturity t is:

        y(t) = Beta0
             + Beta1 * (1 - exp(-t / Tau)) / (t / Tau)
             + Beta2 * [(1 - exp(-t / Tau)) / (t / Tau) - exp(-t / Tau)]

    Parameters
    ----------
    Beta0 : float
        Long-run level of interest rates (asymptote as t -> infinity).
    Beta1 : float
        Short-term component (slope).  Beta0 + Beta1 is the short rate.
    Beta2 : float
        Medium-term component (curvature / hump).
    Tau : float
        Decay factor that controls the position of the hump.
    """

    def __init__(self, initial_guess=None):
        """
        Parameters
        ----------
        initial_guess : list or np.ndarray of length 4, optional
            Starting values for [Beta0, Beta1, Beta2, Tau].
            Defaults to [0.03, -0.02, 0.01, 2.0].
        """
        self.Beta0 = None
        self.Beta1 = None
        self.Beta2 = None
        self.Tau = None
        self._initial_guess = (
            np.array(initial_guess, dtype=float)
            if initial_guess is not None
            else np.array([0.03, -0.02, 0.01, 2.0])
        )
        self._fitted = False
        self._rmse = None

    # ------------------------------------------------------------------
    # Nelson-Siegel spot-rate formula
    # ------------------------------------------------------------------
    @staticmethod
    def nelson_siegel(t, Beta0, Beta1, Beta2, Tau):
        """
        Compute spot rate(s) for maturity t under the Nelson-Siegel model.

        Parameters
        ----------
        t : float or np.ndarray
            Time to maturity in years.  Must be > 0.
        Beta0, Beta1, Beta2, Tau : float
            Nelson-Siegel parameters.

        Returns
        -------
        float or np.ndarray
            Spot rate(s) in decimal form (e.g. 0.03 = 3 %).
        """
        t = np.asarray(t, dtype=float)
        # Guard against t = 0
        t_safe = np.where(t == 0, 1e-12, t)

        factor = (1.0 - np.exp(-t_safe / Tau)) / (t_safe / Tau)
        hump = factor - np.exp(-t_safe / Tau)

        return Beta0 + Beta1 * factor + Beta2 * hump

    # ------------------------------------------------------------------
    # Objective function for the optimizer
    # ------------------------------------------------------------------
    def _sse(self, params, maturities, yields):
        """Sum of squared errors between model and market yields."""
        Beta0, Beta1, Beta2, Tau = params
        model_yields = self.nelson_siegel(maturities, Beta0, Beta1, Beta2, Tau)
        errors = yields - model_yields
        return np.sum(errors ** 2)

    # ------------------------------------------------------------------
    # Fit the curve
    # ------------------------------------------------------------------
    def fit_curve(self, maturities, yields, bounds=None, **minimize_kwargs):
        """
        Fit Nelson-Siegel parameters to market data.

        Parameters
        ----------
        maturities : array-like
            Times to maturity in years (e.g. [1, 2, 3, 5, 10]).
        yields : array-like
            Observed market yields in decimal form (e.g. [0.025, 0.027, ...]).
        bounds : list of tuple, optional
            Bounds for [Beta0, Beta1, Beta2, Tau].
            Defaults to [(0, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 30)].
        **minimize_kwargs
            Extra arguments passed to ``scipy.optimize.minimize``.

        Returns
        -------
        self : YieldCurveFitter
            Fitted instance with populated Beta0, Beta1, Beta2, Tau.
        """
        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)

        if bounds is None:
            bounds = [
                (0.0, 0.2),     # Beta0  – long-run level
                (-0.2, 0.2),    # Beta1  – slope component
                (-0.2, 0.2),    # Beta2  – curvature component
                (0.1, 30.0),    # Tau    – decay factor
            ]

        kwargs = {"method": "L-BFGS-B"}
        kwargs.update(minimize_kwargs)

        result = minimize(
            self._sse,
            self._initial_guess,
            args=(maturities, yields),
            bounds=bounds,
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"Optimisation failed: {result.message}")

        self.Beta0, self.Beta1, self.Beta2, self.Tau = result.x
        self._fitted = True
        self._rmse = np.sqrt(result.fun / len(maturities))

        return self

    # ------------------------------------------------------------------
    # Predict yields for arbitrary maturities
    # ------------------------------------------------------------------
    def predict(self, maturities):
        """
        Compute fitted yields for the given maturities.

        Parameters
        ----------
        maturities : array-like
            Maturities in years.

        Returns
        -------
        np.ndarray
            Fitted yields in decimal form.

        Raises
        ------
        RuntimeError
            If ``fit_curve`` has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_curve() before predict().")
        return self.nelson_siegel(
            np.asarray(maturities, dtype=float),
            self.Beta0, self.Beta1, self.Beta2, self.Tau,
        )

    # ------------------------------------------------------------------
    # Summary & properties
    # ------------------------------------------------------------------
    @property
    def rmse(self):
        """Root-mean-square error of the fit, or None before fitting."""
        return self._rmse

    def summary(self):
        """Return a dict of fitted parameters and fit quality."""
        if not self._fitted:
            raise RuntimeError("Call fit_curve() before summary().")
        return {
            "Beta0": self.Beta0,
            "Beta1": self.Beta1,
            "Beta2": self.Beta2,
            "Tau": self.Tau,
            "RMSE_bps": self._rmse * 10000,
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot_curve(
        self,
        maturities,
        yields,
        title="Nelson-Siegel Yield Curve Fit",
        save_path=None,
        show=False,
    ):
        """
        Plot market yields vs the fitted Nelson-Siegel curve.

        Parameters
        ----------
        maturities : array-like
            Market maturities (years).
        yields : array-like
            Market yields (decimal).
        title : str
            Plot title.
        save_path : str or None
            If provided, the figure is saved to this path.
        show : bool
            If True, display the plot interactively.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_curve() before plot_curve().")

        maturities = np.asarray(maturities, dtype=float)
        yields = np.asarray(yields, dtype=float)

        # Dense grid for the fitted curve
        t_min = max(0.1, maturities.min() * 0.5)
        t_max = maturities.max() * 1.5
        t_fine = np.linspace(t_min, t_max, 300)
        y_fine = self.predict(t_fine)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t_fine, y_fine * 100, "b-", linewidth=2, label="Fitted NS curve")
        ax.scatter(
            maturities,
            yields * 100,
            color="red",
            s=60,
            zorder=5,
            label="Market data",
        )

        # Annotate parameters
        param_str = (
            f"$\\beta_0$ = {self.Beta0:.4f}   "
            f"$\\beta_1$ = {self.Beta1:.4f}   "
            f"$\\beta_2$ = {self.Beta2:.4f}   "
            f"$\\tau$ = {self.Tau:.2f}\n"
            f"RMSE = {self._rmse * 10000:.1f} bps"
        )
        ax.text(
            0.02, 0.98, param_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

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
    # Dummy US Treasury yields (decimal): 1Y, 2Y, 3Y, 5Y, 10Y
    maturities = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
    yields = np.array([0.0420, 0.0405, 0.0395, 0.0380, 0.0370])

    fitter = YieldCurveFitter()
    fitter.fit_curve(maturities, yields)

    print("Nelson-Siegel Fit Summary")
    print("=========================")
    for k, v in fitter.summary().items():
        print(f"  {k}: {v}")

    # Save plot automatically
    fitter.plot_curve(
        maturities,
        yields,
        title="Nelson-Siegel Fit – US Treasury Yields",
        save_path="yield_curve_fit.png",
        show=False,
    )
    print("\nPlot saved to yield_curve_fit.png")
