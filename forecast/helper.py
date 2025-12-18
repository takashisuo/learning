import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

class TimeSeriesHelper:
    @staticmethod
    def to_prophet_format(
        df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        exog_cols=None
    ) -> pd.DataFrame:
        """
        任意のDataFrameをProphet用(ds, y, exog...)形式に変換

        Args:
            df (pd.DataFrame): 入力データ
            date_col (str): 日付列の名前
            value_col (str): 値列の名前
            exog_cols (list or None): 説明変数の列名リスト
        return:
            pd.DataFrame: Prophet用に変換されたデータ
        """
        use_cols = [date_col, value_col]
        if exog_cols is not None:
            use_cols += exog_cols
        df_prophet = df[use_cols].copy()
        df_prophet.rename(columns={date_col: "ds", value_col: "y"}, inplace=True)
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
        return df_prophet

    @staticmethod
    def to_statsforecast_format(
        df: pd.DataFrame,
        date_col: str = "date",
        value_col: str = "value",
        unique_id_col: str = None,
        exog_cols=None,
        default_id: str = "series_1"
    ) -> pd.DataFrame:
        """
        任意のDataFrameをStatsForecast用(unique_id, ds, y, exog...)形式に変換
        unique_id_colが指定されていない場合、全行にdefault_idを設定. ただし, unique_id_colは単一指定しかできないため, 
        複数のID列を結合して一意なIDを作成したい場合は, 事前にDataFrameを加工すること.

        Args:
            df (pd.DataFrame): 入力データ
            date_col (str): 日付列の名前
            value_col (str): 値列の名前
            unique_id_col (str or None): 一意なID列の名前
            exog_cols (list or None): 説明変数の列名リスト
            default_id (str): unique_id_colが指定されていない場合に使用する
        
        return:
            pd.DataFrame: StatsForecast用に変換されたデータ
        """
        use_cols = [date_col, value_col]
        if exog_cols is not None:
            use_cols += exog_cols
        df_stats = df[use_cols].copy()

        df_stats.rename(columns={date_col: "ds", value_col: "y"}, inplace=True)
        df_stats["ds"] = pd.to_datetime(df_stats["ds"])

        if unique_id_col is not None and unique_id_col in df.columns:
            df_stats["unique_id"] = df[unique_id_col]
        else:
            df_stats["unique_id"] = default_id

        cols = ["unique_id", "ds", "y"]
        if exog_cols is not None:
            cols += exog_cols
        df_stats = df_stats[cols]

        return df_stats
    
    @staticmethod
    def run_adf_test(df: pd.DataFrame, col: str, significance: float = 0.05, regression: str = 'c') -> dict:
        """
        指定列に対してADF（Augmented Dickey-Fuller）検定を実行する。

        Args:
            df (pd.DataFrame): 対象データ
            col (str): 検定対象の列名
            significance (float): 有意水準（デフォルトは 0.05）
            regression (str): 回帰式の形。'c'（定数項）, 'ct'（定数+トレンド）, 
                              'ctt'（定数+線形+二次トレンド）, 'n'（なし）から選択

        Returns:
            dict: 検定結果（ADF統計量、p値、ラグ数、観測数、臨界値、定常性判定）
        """
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        series = df[col].dropna().astype(float)
        result = adfuller(series, autolag='AIC', regression=regression)

        adf_stat = result[0]
        p_value = result[1]
        lags = result[2]
        n_obs = result[3]
        crit_vals = result[4]

        # 判定と解釈コメント
        if p_value < significance:
            stationary = True
            interpretation = (
                f"p値 = {p_value:.4f} < {significance} のため、"
                "帰無仮説「単位根がある（非定常である）」を棄却します。\n"
                "この系列は定常であると判断されます。"
            )
        else:
            stationary = False
            interpretation = (
                f"p値 = {p_value:.4f} ≥ {significance} のため、"
                "帰無仮説「単位根がある（非定常である）」を棄却できません。\n"
                "この系列は非定常である可能性が高いです。"
            )

        output = {
            'Column': col,
            'ADF Statistic': adf_stat,
            'p-value': p_value,
            'Lags Used': lags,
            'Number of Observations': n_obs,
            'Critical Values': crit_vals,
            'Regression Type': regression,
            'Stationary': stationary,
            'Interpretation': interpretation
        }

        print(f"\nADF Test Result for '{col}' (regression='{regression}')")
        print("--------------------------------------------------")
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Lags Used: {lags}")
        print(f"Number of Observations: {n_obs}")
        print("Critical Values:")
        for k, v in crit_vals.items():
            print(f"  {k}: {v:.4f}")
        print(f"Stationary: {stationary}")
        print("--------------------------------------------------")
        print(interpretation)
        print("--------------------------------------------------")

        return output
    
    @staticmethod
    def granger_causality_test(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        maxlag: int = 12,
        significance: float = 0.05
    ) -> dict:
        """
        グレンジャー因果性検定を実施し、結果を返す。

        Args:
            df (pd.DataFrame): 
                時系列データ（日時順に並んでいることが前提）
            x_col (str): 
                原因（説明変数）と仮定するカラム名
            y_col (str): 
                結果（被説明変数）と仮定するカラム名
            maxlag (int, optional): 
                検定に使用する最大ラグ次数。デフォルトは 12。
            significance (float, optional): 
                有意水準（棄却判定に利用）。デフォルトは 0.05。

        Returns:
            dict: 各ラグに対する結果を格納した辞書。
                - key: "lag_{n}"
                - value: {"p_value": float, "reject_null": bool, "message": str}

        Raises:
            ValueError: x_col または y_col が DataFrame に存在しない場合。
        """
        print(f"\n=== グレンジャー因果性検定: {x_col} → {y_col} ===")
        print(f"最大ラグ: {maxlag}, 有意水準: {significance}\n")
        print("\n帰無仮説: 「x は y をグレンジャー因果しない（x の過去値は y の予測に寄与しない）」")


        # 入力チェック
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"{x_col} または {y_col} が DataFrame に存在しません。")

        # 欠損値除去
        df_clean = df[[y_col, x_col]].dropna()

        # 検定実施
        test_result = grangercausalitytests(df_clean, maxlag=maxlag, verbose=False)

        results = {}
        for lag, res in test_result.items():
            p_value = res[0]['ssr_ftest'][1]
            reject_null = p_value < significance
            message = (
                f"ラグ {lag}: p値={p_value:.4f} → "
                + ("帰無仮説を棄却（因果関係あり）" if reject_null else "帰無仮説を棄却できない（因果関係なし）")
            )

            print(message)

            results[f"lag_{lag}"] = {
                "p_value": p_value,
                "reject_null": reject_null,
                "message": message
            }

        return results