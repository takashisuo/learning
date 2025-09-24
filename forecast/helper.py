import pandas as pd

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