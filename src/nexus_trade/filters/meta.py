from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from nexus_trade.config.profile import MetaLabelingCfg
    from nexus_trade.core.protocols import XGBClassifierProtocol
    from nexus_trade.tools.calibrator import ProbabilityCalibrator


logger = logging.getLogger(__name__)


def load_meta_model(cfg: MetaLabelingCfg, strategy_name: str) -> XGBClassifierProtocol | None:
    if not cfg.enabled:
        return None

    model_path = Path(f"nexus_trade/strategies/{strategy_name}/models/prod_v1.json")
    if not model_path.exists():
        logger.warning(f"{strategy_name}: Meta model not found | path={model_path}")
        return None

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning(f"{strategy_name}: xgboost not installed; meta-labeling disabled")
        return None

    try:
        model = xgb.XGBClassifier()
        model._estimator_type = "classifier"
        model.load_model(model_path.as_posix())
        return model
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to load meta model | err={error}")
        return None


def load_calibration_model(cfg: MetaLabelingCfg, strategy_name: str) -> ProbabilityCalibrator[object] | None:
    if not cfg.use_calibration:
        return None

    calibration_dir = Path(f"nexus_trade/strategies/{strategy_name}/calibration_models")
    if not calibration_dir.exists():
        logger.warning(f"{strategy_name}: Calibration directory not found | path={calibration_dir}")
        return None

    try:
        from nexus_trade.tools.calibrator import ProbabilityCalibrator
    except ImportError:
        logger.warning(f"{strategy_name}: ProbabilityCalibrator not available")
        return None

    try:
        return ProbabilityCalibrator.load_state(str(calibration_dir))
    except FileNotFoundError:
        logger.warning(f"{strategy_name}: Calibration state not found | path={calibration_dir}")
        return None
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to load calibration model | err={error}")
        return None


def load_features_extractor(cfg: MetaLabelingCfg, strategy_name: str) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
    if not cfg.enabled:
        return None

    try:
        module = importlib.import_module(f"strategies.{strategy_name}.features")
    except Exception as error:
        logger.warning(f"{strategy_name}: Failed to import features module | err={error}")
        return None

    extractor: object = getattr(module, "extract_features", None)
    if not callable(extractor):
        logger.error(...)
        return None
    return cast("Callable[[pd.DataFrame], pd.DataFrame]", extractor)
