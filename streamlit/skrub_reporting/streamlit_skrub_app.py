"""
目的: skrub の TableReport / Cleaner / TableVectorizer / tabular_pipeline を
      Streamlit 上で操作しながら確認する応用アプリ。

特徴:
    - TableReport (HTML + JS) を components.html で iframe 埋め込み表示
    - 任意の CSV をアップロードしても同じ画面で EDA → 前処理 → ベースライン学習まで確認できる
    - 除外列をサイドバーで切り替え、目的変数リークがスコアに与える影響を体感できる

起動:
    streamlit run streamlit_skrub_app.py

動作確認: Python 3.11 / streamlit 1.64.0 / skrub 0.10.1 / scikit-learn 1.9.0 / pandas 2.2.1 (conda env: samurai)
          Python 3.12 / streamlit 1.64.0 / skrub 0.10.1 / scikit-learn 1.9.1 / pandas 3.0.6
"""

from __future__ import annotations

import os
from io import BytesIO

# Windows 11 では wmic が無く、joblib (loky) が CPU コア数取得時に警告のトレースバックを出すため抑止する
# (sklearn / skrub の import より前に、論理コア数 "未満" の値を設定する必要がある)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import pandas as pd
import skrub
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from skrub_basic_sample import compare_dtypes, make_cleaner, target_associations
from table_report_layout import set_distribution_layout
from titanic_loader import LEAKAGE_COLUMNS, TARGET, load_titanic, make_dirty_titanic

REPORT_HEIGHT = 900
LAYOUT_LABELS = {"horizontal": "横 (標準)", "wrap": "折り返し", "vertical": "縦 1 列"}


# ---------------------------------------------------------------------------
# キャッシュ付きの計算処理 (Streamlit は操作のたびにスクリプトを再実行するため)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="データを読み込み中...")
def load_dataset(uploaded_bytes: bytes | None) -> pd.DataFrame:
    if uploaded_bytes is None:
        return load_titanic()
    return pd.read_csv(BytesIO(uploaded_bytes))


@st.cache_data(show_spinner="TableReport を生成中...")
def render_table_report(df: pd.DataFrame, title: str, plot_distributions: bool) -> str:
    """TableReport を HTML 文字列として返す (DataFrame が同じなら再計算しない)。"""
    report = skrub.TableReport(df, title=title, verbose=0, plot_distributions=plot_distributions)
    return report.html()


def embed_html(html: str, height: int) -> None:
    """TableReport は JavaScript を含むため iframe で埋め込む。

    新しい streamlit では st.iframe、それ以前は components.html を使う。
    """
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=True)


@st.cache_data(show_spinner="関連度を計算中...")
def cached_target_associations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return target_associations(df, target)


@st.cache_data(show_spinner="交差検証中...")
def cross_validate_baseline(X: pd.DataFrame, y: pd.Series, task: str) -> tuple[float, float]:
    model = skrub.tabular_pipeline(task)
    if task == "classifier":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scoring = "accuracy"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        scoring = "r2"
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()


def infer_task(y: pd.Series) -> str:
    """目的変数から分類 / 回帰を簡易判定する。"""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
        return "classifier"
    return "regressor"


def describe_vectorizer(tv: skrub.TableVectorizer) -> pd.DataFrame:
    """TableVectorizer が各入力列に適用した変換と、出力列数を一覧化する。"""
    return pd.DataFrame(
        [
            {
                "column": col,
                "transformer": type(tr).__name__,
                "n_output_columns": len(tv.input_to_outputs_[col]),
            }
            for col, tr in tv.transformers_.items()
        ]
    )


# ---------------------------------------------------------------------------
# 画面
# ---------------------------------------------------------------------------
def sidebar() -> tuple[pd.DataFrame, str, list[str], bool]:
    st.sidebar.header("データ設定")
    uploaded = st.sidebar.file_uploader("CSV をアップロード (未指定なら titanic)", type=["csv"])
    df = load_dataset(uploaded.getvalue() if uploaded else None)
    is_titanic = uploaded is None

    columns = df.columns.tolist()
    default_target = columns.index(TARGET) if TARGET in columns else len(columns) - 1
    target = st.sidebar.selectbox("目的変数", columns, index=default_target)

    default_drop = [c for c in LEAKAGE_COLUMNS if c in columns] if is_titanic else []
    drop_cols = st.sidebar.multiselect(
        "学習から除外する列",
        [c for c in columns if c != target],
        default=default_drop,
        help="titanic では boat / body が目的変数リーク列。外すとスコアの変化が確認できます。",
    )
    st.sidebar.caption(f"{df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df, target, drop_cols, is_titanic


def tab_table_report(df: pd.DataFrame) -> None:
    st.subheader("TableReport")
    st.markdown(
        "`skrub.TableReport(df)` の 1 行で、**表の閲覧・列ごとの統計/分布・列間の関連度** を"
        "まとめたインタラクティブなレポートが得られます。"
    )
    col1, col2 = st.columns(2)
    plot_distributions = col1.toggle("列ごとの分布グラフを描画する", value=True, help="列数が多い場合は OFF にすると高速です。")
    layout = col2.radio(
        "Distributions の並び方",
        list(LAYOUT_LABELS),
        index=1,
        format_func=LAYOUT_LABELS.get,
        horizontal=True,
    )
    html = render_table_report(df, "TableReport", plot_distributions)
    embed_html(set_distribution_layout(html, layout), REPORT_HEIGHT)
    st.download_button("HTML をダウンロード", html, file_name="table_report.html", mime="text/html")


def tab_associations(df: pd.DataFrame, target: str) -> None:
    st.subheader("目的変数との関連度 (column_associations)")
    st.markdown(
        "Cramér's V (0〜1) で全列の組み合わせの関連度を計算します。"
        "目的変数と **極端に関連度が高い列** は、リーク (未来の情報・答えそのもの) を疑うサインです。"
    )
    assoc = cached_target_associations(df, target)
    if assoc.empty:
        st.info("関連度を計算できる列がありません。")
        return
    threshold = st.slider("リーク警告のしきい値", 0.5, 1.0, 0.8, 0.05)
    suspicious = assoc.loc[assoc["cramer_v"] >= threshold, "column"].tolist()
    if suspicious:
        st.warning(f"リークの疑い: {suspicious}")
    st.bar_chart(assoc, x="column", y="cramer_v", horizontal=True, sort="-cramer_v")
    st.dataframe(assoc, hide_index=True)


def tab_cleaner(df: pd.DataFrame, is_titanic: bool) -> None:
    st.subheader("Cleaner: 汚いデータの自動整形")
    if is_titanic:
        st.markdown(
            "titanic を意図的に汚したデータ (数値が文字列 / `'N/A'` 表記の欠損 / 日付文字列 / "
            "定数列 / ほぼ欠損の列) に `skrub.Cleaner` を適用した結果です。"
        )
        before = make_dirty_titanic(df)
    else:
        before = df
    after = make_cleaner().fit_transform(before)

    diff = compare_dtypes(before, after)
    diff["changed"] = diff["before"] != diff["after"]
    st.dataframe(diff.sort_values("changed", ascending=False))

    col1, col2 = st.columns(2)
    col1.markdown("**Before**")
    col1.dataframe(before.head(10))
    col2.markdown("**After**")
    col2.dataframe(after.head(10))


def tab_vectorizer(X: pd.DataFrame) -> None:
    st.subheader("TableVectorizer: 列の型に応じた自動エンコード")
    st.markdown(
        "数値はそのまま、低カーディナリティのカテゴリは One-Hot、自由記述の文字列は StringEncoder、"
        "日付は DatetimeEncoder … と、**列ごとに適切な変換を自動で選択** します。"
    )
    tv = skrub.TableVectorizer()
    Xt = tv.fit_transform(X)
    st.metric("特徴量の数", f"{X.shape[1]} 列 → {Xt.shape[1]} 列")
    st.dataframe(describe_vectorizer(tv), hide_index=True)
    with st.expander("変換後のデータ (先頭 20 行)"):
        st.dataframe(Xt.head(20))


def tab_model(df: pd.DataFrame, target: str, drop_cols: list[str]) -> None:
    st.subheader("tabular_pipeline: 前処理を書かずにベースライン学習")
    st.code(
        'model = skrub.tabular_pipeline("classifier")  # TableVectorizer + HistGradientBoosting',
        language="python",
    )

    data = df.dropna(subset=[target])
    y = data[target]
    task = infer_task(y)
    metric = "accuracy" if task == "classifier" else "R²"
    st.caption(f"タスク: {task} / 評価指標: {metric} (5-fold CV)")

    if not st.button("学習・評価する", type="primary"):
        return

    X_all = data.drop(columns=[target])
    col1, col2 = st.columns(2)
    mean, std = cross_validate_baseline(X_all, y, task)
    col1.metric(f"全列を使用 ({X_all.shape[1]} 列)", f"{mean:.3f}", help=f"std={std:.3f}")
    if drop_cols:
        X_dropped = X_all.drop(columns=drop_cols)
        mean_d, std_d = cross_validate_baseline(X_dropped, y, task)
        col2.metric(
            f"除外列あり ({X_dropped.shape[1]} 列)",
            f"{mean_d:.3f}",
            delta=f"{mean_d - mean:+.3f}",
            help=f"除外: {drop_cols} / std={std_d:.3f}",
        )
        st.info("リーク列を含むとスコアは高く見えますが、本番では使えない情報です。除外後のスコアが実力値です。")


def main() -> None:
    st.set_page_config(page_title="skrub × Streamlit", page_icon="🧽", layout="wide")
    st.title("🧽 skrub × Streamlit")
    st.caption("skrub を使った EDA → 前処理 → ベースライン学習 の流れを確認するアプリ")

    df, target, drop_cols, is_titanic = sidebar()
    X = df.drop(columns=[target, *drop_cols])

    tabs = st.tabs(["📊 TableReport", "🔗 関連度", "🧹 Cleaner", "⚙️ TableVectorizer", "🤖 モデル"])
    with tabs[0]:
        tab_table_report(df)
    with tabs[1]:
        tab_associations(df, target)
    with tabs[2]:
        tab_cleaner(df, is_titanic)
    with tabs[3]:
        tab_vectorizer(X)
    with tabs[4]:
        tab_model(df, target, drop_cols)


if __name__ == "__main__":
    main()
