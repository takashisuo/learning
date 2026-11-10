"""Split Conformal Prediction の実装。

任意の scikit-learn 互換回帰モデル(.fit / .predict を持つもの)をラップし、
点予測に対して「有限サンプルで理論的な被覆率(coverage)が保証される」
予測区間を付与する。

アルゴリズム(split conformal prediction, Lei et al. 2018):
1. モデルを学習データ(train)で学習する
2. キャリブレーションデータ(calibration)で非適合度スコア(ここでは残差の絶対値)を計算する
3. 非適合度スコアの (1 - alpha) 分位点(有限サンプル補正込み)を q_hat とする
4. 新規データの点予測 ± q_hat を予測区間とする
"""

from __future__ import annotations

import numpy as np


class SplitConformalRegressor:
    """Split Conformal Prediction による回帰の予測区間推定器。

    Parameters
    ----------
    model: scikit-learn 互換の回帰モデル(fit/predict を持つもの)
    alpha: 有意水準。区間の目標被覆率は (1 - alpha) となる(例: alpha=0.1 -> 90%区間)
    """

    def __init__(self, model, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.model = model
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit(self, X_train, y_train) -> "SplitConformalRegressor":
        """ベースモデルを学習データで学習する。"""
        self.model.fit(X_train, y_train)
        return self

    def calibrate(self, X_calib, y_calib) -> "SplitConformalRegressor":
        """キャリブレーションデータから非適合度スコアの分位点 q_hat を求める。"""
        y_pred = self.model.predict(X_calib)
        residuals = np.abs(np.asarray(y_calib) - y_pred)

        n = len(residuals)
        # 有限サンプル補正: ceil((n+1)(1-alpha)) / n 分位点を取ることで
        # 有限サンプルでも被覆率 >= 1-alpha が保証される
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = float(np.quantile(residuals, q_level, method="higher"))
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """点予測と予測区間(下限・上限)を返す。"""
        if self.q_hat is None:
            raise RuntimeError("calibrate() must be called before predict()")

        y_pred = self.model.predict(X)
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return y_pred, lower, upper

    def coverage(self, y_true, lower, upper) -> float:
        """実際の被覆率(区間内に真値が入った割合)を計算する。"""
        y_true = np.asarray(y_true)
        inside = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(inside))
