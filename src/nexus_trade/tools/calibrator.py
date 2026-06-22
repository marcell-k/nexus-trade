from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np

try:
    import joblib  # pyright: ignore[reportMissingImports]

except ImportError as _ml_import_err:
    raise ImportError(
        "calibrator.py requires optional ML deps: pip install joblib scipy scikit-learn"
    ) from _ml_import_err


logger = logging.getLogger(__name__)


class SklearnClassifierProtocol(Protocol):
    """Binary sklearn classifier operating on ndarray inputs."""

    coef_: np.ndarray
    intercept_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class IsotonicModelProtocol(Protocol):
    """Isotonic regression operating on ndarray inputs."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


CalibratorModel = SklearnClassifierProtocol | IsotonicModelProtocol | Literal["temperature"]


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
        self.model: CalibratorModel | None = None
        self.T: float | None = None
        self._fitted: bool = False

        # Meta-method attributes
        self.base_calibrators: dict[str, ProbabilityCalibrator] = {}
        self.meta_model: SklearnClassifierProtocol | None = None

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
            _m = cast("SklearnClassifierProtocol", self.model)
            return _m.predict_proba(self._logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            _m_iso = cast("IsotonicModelProtocol", self.model)
            return _m_iso.predict(p)
        if self.method == "beta":
            _m = cast("SklearnClassifierProtocol", self.model)
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
