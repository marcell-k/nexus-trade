from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from nexus_trade.core.protocols import ClassifierWithProba

try:
    import joblib  # pyright: ignore[reportMissingImports]
    from scipy.optimize import minimize  # pyright: ignore[reportMissingImports]
    from sklearn.isotonic import IsotonicRegression  # pyright: ignore[reportMissingImports]
    from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
    from sklearn.metrics import brier_score_loss, log_loss  # pyright: ignore[reportMissingImports]

except ImportError as _ml_import_err:
    raise ImportError(
        "calibrator.py requires optional ML deps: pip install joblib scipy scikit-learn"
    ) from _ml_import_err


logger = logging.getLogger(__name__)


class _SklearnClassifier(Protocol):
    """Binary sklearn classifier operating on ndarray inputs."""

    coef_: np.ndarray
    intercept_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class _IsotonicModel(Protocol):
    """Isotonic regression operating on ndarray inputs."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


_ModelType = _SklearnClassifier | _IsotonicModel | Literal["temperature"]


class ProbabilityCalibrator:
    r"""
    Post-hoc probability calibration with proper train/test separation.

    Supports base methods (Platt, Beta, Isotonic, Temperature) and meta-methods (Ensemble, Stacking).

    Formulations:
    - Platt: logit(p') = a * logit(p) + b
    - Isotonic: Non-parametric monotone mapping
    - Beta: logit(p') = a * logit(p) + b * logit(1-p) + c
    - Temperature: p' = sigmoid(logit(p) / T)
    - Ensemble: p' = (1/K) * Σ g_k(p) where g_k are base calibrators
    - Stacking: p' = σ(Σ w_k * logit(g_k(p)) + b) with learned weights w_k

    All calibration fits are trained on past-only data and evaluated on strictly later data
    to avoid leakage in systematic trading.

    """  # noqa: RUF002

    def __init__(
        self, method: str = "platt", base_methods: tuple[str, ...] = ("platt", "beta", "isotonic", "temperature")
    ) -> None:
        self.method: str = method.lower()
        self.base_methods: tuple[str, ...] = base_methods
        self.model: _ModelType | None = None
        self.T: float | None = None
        self._fitted: bool = False

        # Meta-method attributes
        self.base_calibrators: dict[str, ProbabilityCalibrator] = {}
        self.meta_model: _SklearnClassifier | None = None

        # Validation
        valid_base_methods = {"platt", "beta", "isotonic", "temperature"}
        valid_meta_methods = {"ensemble", "stacking"}
        all_valid = valid_base_methods | valid_meta_methods

        if self.method not in all_valid:
            raise ValueError(f"method must be one of: {all_valid}")

        # Validate base_methods for meta-methods
        if self.method in valid_meta_methods:
            invalid = set(base_methods) - valid_base_methods
            if invalid:
                raise ValueError(f"base_methods contains invalid methods: {invalid}")

    @staticmethod
    def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Compute logit with numerical stability clipping."""
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    @staticmethod
    def _score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
        """Return (brier_score, log_loss) for predictions."""
        return brier_score_loss(y, p), log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))

    def fit(
        self,
        y_cal: np.ndarray,
        p_cal: np.ndarray,
        y_dev: np.ndarray | None = None,
        p_dev: np.ndarray | None = None,
    ) -> ProbabilityCalibrator:
        """Fit calibration mapping. Stacking requires y_dev/p_dev for meta-model."""
        y_cal = np.asarray(y_cal, dtype=int).reshape(-1)
        p_cal = np.asarray(p_cal, dtype=float).reshape(-1)

        if y_cal.shape[0] != p_cal.shape[0]:
            raise ValueError("y_cal and p_cal must have same length")

        # Branch: Meta-methods (ensemble, stacking)
        if self.method in {"ensemble", "stacking"}:
            return self._fit_meta(y_cal, p_cal, y_dev, p_dev)

        # Branch: Base methods
        return self._fit_base(y_cal, p_cal)

    def _fit_base(self, y_cal: np.ndarray, p_cal: np.ndarray) -> ProbabilityCalibrator:
        """Fit base calibration method (platt, beta, isotonic, temperature)."""
        if self.method == "platt":
            X = self._logit(p_cal).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, solver="lbfgs", random_state=42, max_iter=1000)
            lr.fit(X, y_cal)
            self.model = lr

        elif self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(p_cal, y_cal)
            self.model = iso

        elif self.method == "beta":
            X = np.column_stack([self._logit(p_cal), self._logit(1 - p_cal)])
            lr = LogisticRegression(C=1e6, solver="lbfgs", random_state=42, max_iter=1000)
            lr.fit(X, y_cal)
            self.model = lr

        elif self.method == "temperature":
            z_cal = self._logit(p_cal)

            def nll(tlog: np.ndarray) -> float:
                T = float(np.exp(tlog[0]))
                p = 1.0 / (1.0 + np.exp(-z_cal / T))
                return log_loss(y_cal, np.clip(p, 1e-12, 1 - 1e-12))

            res = minimize(nll, x0=[0.0], method="L-BFGS-B", bounds=[(-5, 5)])
            self.T = float(np.exp(res.x[0]))
            self.model = "temperature"

        self._fitted = True
        return self

    def _fit_meta(
        self,
        y_cal: np.ndarray,
        p_cal: np.ndarray,
        y_dev: np.ndarray | None,
        p_dev: np.ndarray | None,
    ) -> ProbabilityCalibrator:
        r"""
        Fit meta-calibration method (ensemble or stacking).

        Ensemble: Fit all base calibrators on train data (y_cal, p_cal).
        Stacking: Fit base calibrators on train, meta-model on dev.

        Mathematical formulation for stacking meta-model:
        $$
        p_{\\text{stack}} = \\sigma\\left(\\sum_{k=0}^{K} w_k \\cdot \\text{logit}(p_k) + b\\right)
        $$
        where \\(p_0\\) is original probability, \\(p_k\\) are base calibrator outputs,
        and \\(w_k, b\\) are learned via logistic regression on dev set.
        """
        # Validation: Stacking requires dev data
        if self.method == "stacking":
            if y_dev is None or p_dev is None:
                raise ValueError(
                    "Stacking calibration requires dev data (y_dev, p_dev) for meta-model training. "
                    "Pass y_dev and p_dev to fit() or use 'ensemble' method instead."
                )
            y_dev = np.asarray(y_dev, dtype=int).reshape(-1)
            p_dev = np.asarray(p_dev, dtype=float).reshape(-1)
            if y_dev.shape[0] != p_dev.shape[0]:
                raise ValueError("y_dev and p_dev must have same length")

        # Stage 1: Fit base calibrators on train data
        logger.info(f"CalibFit method={self.method}")
        logger.info(f"CalibFitStage1 n_base={len(self.base_methods)} | n_cal={len(y_cal)}")

        for base_method in self.base_methods:
            try:
                calibrator = ProbabilityCalibrator(method=base_method)
                calibrator.fit(y_cal, p_cal)
                self.base_calibrators[base_method] = calibrator
                logger.debug(f"CalibBaseFitOK method={base_method}")
            except Exception as e:
                logger.warning(f"CalibBaseFitFail method={base_method} | err={e}")

        if not self.base_calibrators:
            raise ValueError("All base calibrators failed to fit")

        # Stage 2 (Stacking only): Fit meta-model on dev predictions
        if self.method == "stacking":
            assert y_dev is not None
            assert p_dev is not None
            logger.info(f"CalibFitStage2 n_dev={len(y_dev)}")

            # Generate meta-features: logit-transformed probabilities from each calibrator
            # Feature matrix shape: (n_dev, K+1) where K = num base calibrators
            # Features: [logit(p_original), logit(p_platt), logit(p_beta), ...]
            meta_features_dev = [self._logit(p_dev)]  # Original probability as baseline feature

            for base_method, calibrator in self.base_calibrators.items():
                try:
                    p_cal_dev = calibrator.transform(p_dev)
                    meta_features_dev.append(self._logit(p_cal_dev))
                except Exception as e:
                    logger.warning(f"CalibTransformFail method={base_method} | stage=dev | err={e}")

            X_meta_dev = np.column_stack(meta_features_dev)

            # Fit logistic regression meta-model
            # C=1.0 provides L2 regularization to prevent overfitting on dev set
            _meta = LogisticRegression(C=1.0, solver="lbfgs", random_state=42, max_iter=1000)
            _meta.fit(X_meta_dev, y_dev)
            self.meta_model = _meta

            # Print learned weights (diagnostic)
            feature_names = ["original", *list(self.base_calibrators.keys())]
            weight_str = ", ".join(f"{n}={w:+.4f}" for n, w in zip(feature_names, _meta.coef_[0], strict=True))
            logger.debug(f"CalibMetaWeights {weight_str} | intercept={_meta.intercept_[0]:+.4f}")

        logger.info(f"CalibFitOK method={self.method}")
        self._fitted = True
        return self

    def transform(self, p_infer: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to uncalibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before transform")

        p = np.asarray(p_infer, dtype=float).reshape(-1)

        if self.method == "ensemble":
            return self._transform_ensemble(p)
        if self.method == "stacking":
            return self._transform_stacking(p)
        if self.model is None:
            raise RuntimeError(f"Model not fitted for method '{self.method}'")
        if self.method == "platt":
            _m = cast("_SklearnClassifier", self.model)
            return _m.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            _m_iso = cast("_IsotonicModel", self.model)
            return _m_iso.predict(p)
        if self.method == "beta":
            _m = cast("_SklearnClassifier", self.model)
            X = np.column_stack([self._logit(p), self._logit(1 - p)])
            return _m.predict_proba(X)[:, 1]
        if self.method == "temperature":
            if self.T is None:
                raise RuntimeError("Temperature T not set")
            z = self._logit(p)
            return 1.0 / (1.0 + np.exp(-z / self.T))
        raise RuntimeError(f"Unknown base method: '{self.method}'")

    def _transform_ensemble(self, p: np.ndarray) -> np.ndarray:
        r"""
        Ensemble transform: Average predictions from all base calibrators.

        Mathematical formulation:
        $$
        p_{\\text{ensemble}} = \\frac{1}{K} \\sum_{k=1}^{K} g_k(p)
        $$
        where \\(g_k\\) are base calibrators.
        """
        calibrated_predictions = []

        for base_method, calibrator in self.base_calibrators.items():
            try:
                p_cal = calibrator.transform(p)
                calibrated_predictions.append(p_cal)
            except Exception as e:
                logger.warning(f"CalibTransformFail method={base_method} | err={e}")

        if not calibrated_predictions:
            raise RuntimeError("All base calibrators failed during transform")

        # Vectorized averaging: O(K * n) memory, O(n) final operation
        return np.mean(calibrated_predictions, axis=0)

    def _transform_stacking(self, p: np.ndarray) -> np.ndarray:
        r"""
        Stacking transform: Apply meta-model to base calibrators' predictions.

        Mathematical formulation:
        $$
        p_{\\text{stack}} = \\sigma\\left(\\sum_{k=0}^{K} w_k \\cdot \\text{logit}(p_k) + b\\right)
        $$
        """
        # Generate meta-features (same structure as training)
        meta_features = [self._logit(p)]

        for base_method, calibrator in self.base_calibrators.items():
            try:
                p_cal = calibrator.transform(p)
                meta_features.append(self._logit(p_cal))
            except Exception as e:
                logger.warning(f"CalibTransformFail method={base_method} | stage=stacking | err={e}")
                # Fallback: use original probability to maintain feature dimensionality
                meta_features.append(self._logit(p))

        X_meta = np.column_stack(meta_features)

        # Meta-model prediction: O(n * K) complexity
        if self.meta_model is None:
            raise RuntimeError("Meta-model not fitted")
        return self.meta_model.predict_proba(X_meta)[:, 1]

    @staticmethod
    def calculate_prob_array(X: pd.DataFrame, final_model: ClassifierWithProba) -> np.ndarray:
        """Calculate model probabilities as float32 numpy array. Drops index/time info."""
        if X.empty:
            raise ValueError("Feature DataFrame X is empty")

        feature_cols = [col for col in X.columns if col not in ["EntryTime", "ExitTime"]]
        if not feature_cols:
            raise ValueError("No feature columns available for prediction")

        return final_model.predict_proba(X[feature_cols])[:, 1].astype(np.float32)

    @classmethod
    def _compute_split_probs(
        cls, X_train: pd.DataFrame, X_dev: pd.DataFrame, X_test: pd.DataFrame, final_model: ClassifierWithProba
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute probability arrays for train/dev/test splits."""
        return (
            cls.calculate_prob_array(X_train, final_model),
            cls.calculate_prob_array(X_dev, final_model),
            cls.calculate_prob_array(X_test, final_model),
        )

    @classmethod
    def select_best_calibration(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_dev: pd.DataFrame,
        y_dev: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        final_model: ClassifierWithProba,
        methods: tuple[str, ...] = (
            "platt",
            "beta",
            "isotonic",
            "temperature",
            "ensemble",
            "stacking",
        ),
    ) -> tuple[list[tuple[str, ...]], str]:
        """Select best calibration method by dev Brier score across all splits."""
        all_scores = []

        p_train, p_dev, p_test = cls._compute_split_probs(X_train, X_dev, X_test, final_model)

        # Standardize label arrays
        y_train = np.asarray(y_train, dtype=int).reshape(-1)
        y_dev = np.asarray(y_dev, dtype=int).reshape(-1)
        y_test = np.asarray(y_test, dtype=int).reshape(-1)

        # Validation: length alignment
        if y_train.shape[0] != len(p_train) or y_dev.shape[0] != len(p_dev) or y_test.shape[0] != len(p_test):
            raise ValueError("y arrays must have the same length as corresponding X rows")

        # Baseline scores (original probabilities)
        bs_train, ll_train = cls._score(y_train, p_train)
        bs_dev, ll_dev = cls._score(y_dev, p_dev)
        bs_test, ll_test = cls._score(y_test, p_test)
        all_scores.append(("original", bs_train, ll_train, bs_dev, ll_dev, bs_test, ll_test))

        # Test calibration methods
        for method in methods:
            try:
                calibrator = cls(method)

                # Fit: Pass dev data for stacking, ignored for others
                calibrator.fit(y_train, p_train, y_dev, p_dev)

                # Apply calibration to all splits
                p_train_cal = calibrator.transform(p_train)
                p_dev_cal = calibrator.transform(p_dev)
                p_test_cal = calibrator.transform(p_test)

                # Calculate scores for all splits
                bs_train_cal, ll_train_cal = cls._score(y_train, p_train_cal)
                bs_dev_cal, ll_dev_cal = cls._score(y_dev, p_dev_cal)
                bs_test_cal, ll_test_cal = cls._score(y_test, p_test_cal)
                all_scores.append(
                    (
                        method,
                        bs_train_cal,
                        ll_train_cal,
                        bs_dev_cal,
                        ll_dev_cal,
                        bs_test_cal,
                        ll_test_cal,
                    )
                )

            except Exception as e:
                logger.warning(f"CalibSelectFail method={method} | err={e}")

        # Sort by validation Brier score (index 3)
        best_method = min(all_scores, key=lambda x: x[3])[0]

        return all_scores, best_method

    @staticmethod
    def print_calibration_scores(scores: list[tuple[str, ...]]) -> None:
        """Print calibration scores in table format."""
        header = (
            f"{'Method':<12} {'Train Brier':>12} {'Train LogLoss':>14} {'Dev Brier':>12}"
            f"{'Dev LogLoss':>14} {'Test Brier':>12} {'Test LogLoss':>14}"
        )
        lines = ["=" * 95, "Calibration Scores (Proper Train/Dev/Test Validation)", "-" * 95, header, "-" * 95]
        for row in scores:
            method, bs_train, ll_train, bs_dev, ll_dev, bs_test, ll_test = row
            lines.append(
                f"{method:<12} {bs_train:>12.6f} {ll_train:>14.6f} {bs_dev:>12.6f} "
                f"{ll_dev:>14.6f} {bs_test:>12.6f} {ll_test:>14.6f}"
            )
        lines.append("=" * 95)
        logger.info("\n".join(lines))

    @classmethod
    def integrate_calibrated_probs(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_dev: pd.DataFrame,
        X_test: pd.DataFrame,
        final_model: ClassifierWithProba,
        method: str = "isotonic",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply calibration and return probability arrays for train, dev, and test sets."""
        prob_train, prob_dev, prob_test = cls._compute_split_probs(X_train, X_dev, X_test, final_model)

        if method.lower() == "original":
            logger.info("CalibIntegrate method=original | action=skip")
            return prob_train, prob_dev, prob_test
        else:
            # Fit calibrator
            calibrator = cls(method)

            # For stacking, pass dev data; for others, dev data is ignored
            y_train_arr = np.asarray(y_train, dtype=int).reshape(-1)

            if method.lower() == "stacking":
                logger.warning(
                    "CalibIntegrate method=stacking | y_dev=missing | action=train_only | "
                    "prefer=select_best_calibration or full_calibration_workflow"
                )
                calibrator.fit(y_train_arr, prob_train)
            else:
                calibrator.fit(y_train_arr, prob_train)

            # Apply calibration to all probability sets
            cal_probs_train = calibrator.transform(prob_train)
            cal_probs_dev = calibrator.transform(prob_dev)
            cal_probs_test = calibrator.transform(prob_test)

            logger.info(f"CalibIntegrate method={method} | ok=1")
            return cal_probs_train, cal_probs_dev, cal_probs_test

    @classmethod
    def full_calibration_workflow(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_dev: pd.DataFrame,
        y_dev: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        final_model: ClassifierWithProba,
        methods: tuple[str, ...] = (
            "platt",
            "beta",
            "isotonic",
            "temperature",
            "ensemble",
            "stacking",
        ),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[tuple]]:
        """Select best method, apply calibration, return (probs_train, probs_dev, probs_test, best_method, scores)."""
        # Select best calibration method
        calibration_scores, best_method = cls.select_best_calibration(
            X_train, y_train, X_dev, y_dev, X_test, y_test, final_model, methods
        )

        logger.info(f"CalibSelect best={best_method}")
        cls.print_calibration_scores(calibration_scores)

        # Apply best calibration method to generate probabilities
        logger.info(f"CalibApply method={best_method}")

        p_train, p_dev, p_test = cls._compute_split_probs(X_train, X_dev, X_test, final_model)

        if best_method == "original":
            probs_train, probs_dev, probs_test = p_train, p_dev, p_test
        else:
            # Fit calibrator with proper train/dev separation
            calibrator = cls(best_method)
            y_train_arr = np.asarray(y_train, dtype=int).reshape(-1)
            y_dev_arr = np.asarray(y_dev, dtype=int).reshape(-1)

            calibrator.fit(y_train_arr, p_train, y_dev_arr, p_dev)

            probs_train = calibrator.transform(p_train)
            probs_dev = calibrator.transform(p_dev)
            probs_test = calibrator.transform(p_test)

        return probs_train, probs_dev, probs_test, best_method, calibration_scores

    @classmethod
    def apply_calibration_with_method(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_dev: pd.DataFrame,
        X_test: pd.DataFrame,
        final_model: ClassifierWithProba,
        method: str,
        y_dev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply specified calibration method (no selection step). y_dev required for stacking."""
        prob_train, prob_dev, prob_test = cls._compute_split_probs(X_train, X_dev, X_test, final_model)

        if method.lower() == "original":
            logger.info("CalibApply method=original | action=skip")
            return prob_train, prob_dev, prob_test
        else:
            # Apply specified calibration method
            calibrator = cls(method)
            y_train_arr = np.asarray(y_train, dtype=int).reshape(-1)

            if method.lower() == "stacking":
                if y_dev is None:
                    raise ValueError("Stacking calibration requires y_dev parameter")
                y_dev_arr = np.asarray(y_dev, dtype=int).reshape(-1)
                calibrator.fit(y_train_arr, prob_train, y_dev_arr, prob_dev)
            else:
                calibrator.fit(y_train_arr, prob_train)

            # Transform all probability sets
            cal_probs_train = calibrator.transform(prob_train)
            cal_probs_dev = calibrator.transform(prob_dev)
            cal_probs_test = calibrator.transform(prob_test)

            logger.info(f"CalibApply method={method} | ok=1")
            return cal_probs_train, cal_probs_dev, cal_probs_test

    @classmethod
    def fit_production_calibrator(
        cls,
        X_full: pd.DataFrame,
        y_full: np.ndarray,
        production_model: ClassifierWithProba,
        method: str = "platt",
        save_path: str | None = None,
    ) -> ProbabilityCalibrator:
        """Fit production calibrator on 100% of data. Stacking uses 70/30 temporal split."""
        logger.info(f"CalibProdFit method={method} | n={len(X_full)}")

        # Standardize labels
        y_full = np.asarray(y_full, dtype=int).reshape(-1)

        if len(y_full) != len(X_full):
            raise ValueError(f"y_full length ({len(y_full)}) must match X_full rows ({len(X_full)})")

        # Get uncalibrated probabilities from production model on full data
        p_full_uncalibrated = cls.calculate_prob_array(X_full, production_model)

        # Compute baseline metrics
        bs_uncalibrated, ll_uncalibrated = cls._score(y_full, p_full_uncalibrated)

        logger.info(f"CalibProdBase brier={bs_uncalibrated:.6f} | log_loss={ll_uncalibrated:.6f}")

        # Special handling for stacking: needs train/dev split
        if method.lower() == "stacking":
            split_idx = int(len(y_full) * 0.7)

            y_train_prod = y_full[:split_idx]
            p_train_prod = p_full_uncalibrated[:split_idx]
            y_dev_prod = y_full[split_idx:]
            p_dev_prod = p_full_uncalibrated[split_idx:]

            calibrator = cls(method=method)
            calibrator.fit(y_train_prod, p_train_prod, y_dev_prod, p_dev_prod)
        else:
            # Ensemble and base methods: use 100% of data
            logger.info(f"CalibProdFit method={method} | scope=full")
            calibrator = cls(method=method)
            calibrator.fit(y_full, p_full_uncalibrated)

        # Compute calibrated metrics (for verification only)
        p_full_calibrated = calibrator.transform(p_full_uncalibrated)
        bs_calibrated, ll_calibrated = cls._score(y_full, p_full_calibrated)

        improvement = (bs_uncalibrated - bs_calibrated) / bs_uncalibrated * 100
        logger.info(
            f"CalibProdMetrics brier={bs_calibrated:.6f} | log_loss={ll_calibrated:.6f} | "
            f"brier_improvement={improvement:.2f}%"
        )
        # Save calibrator if path provided
        if save_path:
            with Path(save_path).open("wb") as f:
                pickle.dump(calibrator, f)
            logger.info(f"CalibProdSave path={save_path}")

        logger.info(f"CalibProdSummary method={method} | n={len(y_full):,} | pos_rate={np.mean(y_full):.2%}")

        return calibrator

    def save_state(self, output_dir: str = "models") -> None:
        """Save calibrator state (JSON) and models (joblib) to directory."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Build state dictionary
        state = {
            "method": self.method,
            "base_methods": list(self.base_methods) if self.base_methods else None,
            "fitted": self._fitted,
            "fitted_models": {},
        }

        # Save models
        if self.method in {"ensemble", "stacking"}:
            # Save each base calibrator
            for name, base_cal in self.base_calibrators.items():
                logger.debug(f"CalibSave name={name}")

                if base_cal.model is not None and base_cal.method != "temperature":
                    model_path = Path(output_dir) / f"{name}_model.joblib"
                    joblib.dump(base_cal.model, model_path)
                    logger.debug(f"CalibSaveModel path={model_path}")

                if hasattr(base_cal, "T") and base_cal.T is not None:
                    state["fitted_models"][name] = {"T": float(base_cal.T)}
                    logger.debug(f"CalibSaveTemp name={name} | T={base_cal.T:.4f}")

            # Save stacking meta-model
            if self.method == "stacking" and self.meta_model is not None:
                meta_path = Path(output_dir) / "stacking_meta_model.joblib"
                joblib.dump(self.meta_model, meta_path)
                logger.debug(f"CalibSaveMetaModel path={meta_path}")

        else:
            # Save single base method model
            if self.model is not None and self.method != "temperature":
                model_path = Path(output_dir) / f"{self.method}_model.joblib"
                joblib.dump(self.model, model_path)
                logger.debug(f"CalibSaveModel method={self.method} | path={model_path}")

            if hasattr(self, "T") and self.T is not None:
                state["T"] = float(self.T)
                logger.debug(f"CalibSaveTemp T={self.T:.4f}")

        # Save state JSON
        state_path = Path(output_dir) / "calibrator_state.json"
        state_path.write_text(json.dumps(state, indent=2))

        n_files = len(self.base_calibrators) if self.method in {"ensemble", "stacking"} else 1
        logger.info(f"CalibSaveDone dir={output_dir} | json=1 | models={n_files}")

    @classmethod
    def load_state(cls, input_dir: str = "models") -> ProbabilityCalibrator:
        """Load calibrator from state directory (JSON config + joblib models)."""
        # Load state
        state_path = Path(input_dir) / "calibrator_state.json"
        if not Path(state_path).exists():
            raise FileNotFoundError(f"Calibrator state not found: {state_path}")

        with Path.open(state_path) as f:
            state = json.load(f)

        # Reconstruct calibrator
        calibrator = cls(
            method=state["method"],
            base_methods=tuple(state["base_methods"]) if state["base_methods"] else (),
        )
        calibrator._fitted = state.get("fitted", True)

        # Load models
        if state["method"] in {"ensemble", "stacking"}:
            # Load base calibrators
            for name in state["base_methods"]:
                base_cal = cls(method=name)

                # Load sklearn model
                model_path = Path(input_dir) / f"{name}_model.joblib"
                if Path.exists(model_path):
                    base_cal.model = joblib.load(model_path)

                # Load temperature
                if name in state.get("fitted_models", {}) and "T" in state["fitted_models"][name]:
                    base_cal.T = state["fitted_models"][name]["T"]
                    base_cal.model = "temperature"

                base_cal._fitted = True
                calibrator.base_calibrators[name] = base_cal

            # Load stacking meta-model
            if state["method"] == "stacking":
                meta_path = Path(input_dir) / "stacking_meta_model.joblib"
                if Path.exists(meta_path):
                    calibrator.meta_model = joblib.load(meta_path)

        else:
            # Load single base method
            model_path = Path(input_dir) / f"{state['method']}_model.joblib"
            if Path(model_path).exists():
                calibrator.model = joblib.load(model_path)
            if "T" in state:
                calibrator.T = state["T"]
                calibrator.model = "temperature"

        return calibrator
