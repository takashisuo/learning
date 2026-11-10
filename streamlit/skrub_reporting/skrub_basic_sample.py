"""
目的: skrub の主なメリットを、titanic データを使って最小限のコードで確認するサンプル。

確認できるメリット:
    1. TableReport        : 1 行でインタラクティブな EDA レポート (HTML) を作れる
    2. column_associations: 列間の関連度を一覧化し、目的変数リークを発見できる
    3. Cleaner            : 「汚い」データの型・欠損表現・不要列を自動で整える
    4. tabular_pipeline   : 列ごとの前処理を書かずに、そのまま学習できるベースラインを作れる

使い方:
    python skrub_basic_sample.py            # レポートを output/ に保存
    python skrub_basic_sample.py --open     # 保存後にブラウザで開く
    python skrub_basic_sample.py --layout vertical  # Distributions を縦 1 列で表示 (horizontal / wrap / vertical)

動作確認: Python 3.11 / skrub 0.10.1 / scikit-learn 1.9.0 / pandas 2.2.1 (conda env: samurai)
          Python 3.12 / skrub 0.10.1 / scikit-learn 1.9.1 / pandas 3.0.6
"""

from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

# Windows 11 では wmic が無く、joblib (loky) が CPU コア数取得時に警告のトレースバックを出すため抑止する
# (sklearn / skrub の import より前に、論理コア数 "未満" の値を設定する必要がある)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import pandas as pd
import skrub
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from table_report_layout import Layout, set_distribution_layout
from titanic_loader import LEAKAGE_COLUMNS, TARGET, load_titanic, make_dirty_titanic

OUTPUT_DIR = Path(__file__).parent / "output"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. TableReport: 1 行で EDA レポート
# ---------------------------------------------------------------------------
def build_table_report(
    df: pd.DataFrame, output_dir: Path, open_browser: bool, layout: Layout = "horizontal"
) -> Path:
    """TableReport を作成して HTML に保存する。

    レポートには 表の閲覧 / 列ごとの統計・分布 / 列間の関連度 がタブで含まれる。
    Jupyter であれば `skrub.TableReport(df)` をセルの最後に書くだけで表示される。
    layout で Distributions タブの分布カードの並び方 (横 / 折り返し / 縦) を変更できる。
    """
    report = skrub.TableReport(df, title="Titanic TableReport", verbose=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"titanic_table_report_{layout}.html"
    path.write_text(set_distribution_layout(report.html(), layout), encoding="utf-8")
    print(f"TableReport を保存しました: {path}")
    if open_browser:
        webbrowser.open(path.as_uri())
    return path


# ---------------------------------------------------------------------------
# 2. column_associations: 目的変数リークの発見
# ---------------------------------------------------------------------------
def target_associations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """目的変数と各列の関連度 (Cramér's V) を降順で返す。"""
    assoc = skrub.column_associations(df)
    mask = (assoc["left_column_name"] == target) | (assoc["right_column_name"] == target)
    assoc = assoc[mask].copy()
    assoc["column"] = assoc["left_column_name"].where(
        assoc["left_column_name"] != target, assoc["right_column_name"]
    )
    return assoc[["column", "cramer_v"]].sort_values("cramer_v", ascending=False)


# ---------------------------------------------------------------------------
# 3. Cleaner: 汚いデータの自動整形
# ---------------------------------------------------------------------------
def make_cleaner() -> skrub.Cleaner:
    """数値文字列の変換・欠損文字列の NaN 化・日付変換・定数列/ほぼ欠損列の削除を行う Cleaner。"""
    return skrub.Cleaner(
        parse_numbers=True,  # "71.28" -> 71.28
        drop_if_constant=True,  # 全行同じ値の列を削除
        drop_null_fraction=0.95,  # 95% 以上欠損の列を削除
    )


def compare_dtypes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Cleaner 前後の dtype を並べて比較する。"""
    return pd.DataFrame(
        {
            "before": before.dtypes.astype(str),
            "after": after.dtypes.astype(str).reindex(before.columns).fillna("(削除)"),
        }
    )


# ---------------------------------------------------------------------------
# 4. tabular_pipeline: 前処理を書かずにベースライン作成
# ---------------------------------------------------------------------------
def build_sklearn_pipeline():
    """比較用: scikit-learn だけで組む場合。列名・型ごとの前処理を人手で定義する必要がある。"""
    numeric = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical = ["sex", "embarked"]
    # name / ticket / cabin / home.dest のような文字列列は扱いが難しく、ここでは捨てている
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical,
            ),
        ]
    )
    return make_pipeline(preprocess, HistGradientBoostingClassifier())


def build_skrub_pipeline():
    """skrub: 推定器を渡すだけ。列の型判定・エンコード (文字列列も含む) は TableVectorizer が自動で行う。"""
    return skrub.tabular_pipeline("classifier")


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # 元データは pclass 順に並んでいるためシャッフル
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--open", action="store_true", help="TableReport をブラウザで開く")
    parser.add_argument(
        "--layout",
        choices=["horizontal", "wrap", "vertical"],
        default="horizontal",
        help="Distributions タブの分布カードの並び方 (横 / 折り返し / 縦)",
    )
    args = parser.parse_args()

    df = load_titanic()

    section("1. TableReport: 1 行で EDA レポート")
    build_table_report(df, OUTPUT_DIR, args.open, args.layout)

    section("2. column_associations: 目的変数との関連度")
    assoc = target_associations(df, TARGET)
    print(assoc.head(8).to_string(index=False))
    print(
        f"\n-> {LEAKAGE_COLUMNS} は生存結果を知らないと作れない列 (救命ボート番号/遺体番号)。"
        "\n   関連度が極端に高い列は目的変数リークを疑うきっかけになる。"
    )

    section("3. Cleaner: 汚いデータの自動整形")
    dirty = make_dirty_titanic(df)
    cleaned = make_cleaner().fit_transform(dirty)
    print(compare_dtypes(dirty, cleaned).to_string())
    print(f"\nage の 'N/A' 件数: {(dirty['age'] == 'N/A').sum()} -> 欠損 {cleaned['age'].isna().sum()}")

    section("4. tabular_pipeline: 前処理を書かずにベースライン")
    X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS])
    y = df[TARGET]
    print(f"scikit-learn (前処理を手書き, 7 列のみ利用): {evaluate(build_sklearn_pipeline(), X, y):.3f}")
    print(f"skrub.tabular_pipeline (全 {X.shape[1]} 列を自動処理)  : {evaluate(build_skrub_pipeline(), X, y):.3f}")

    # 参考: リーク列を含めたまま学習すると不自然に高いスコアになる
    X_leak = df.drop(columns=[TARGET])
    print(f"(参考) リーク列を含めた場合               : {evaluate(build_skrub_pipeline(), X_leak, y):.3f}")

    # TableVectorizer が各列をどう処理したかも確認できる
    tv = skrub.TableVectorizer().fit(X)
    print("\nTableVectorizer が自動で選んだ列ごとの変換:")
    for col, transformer in tv.transformers_.items():
        print(f"  {col:<10} -> {transformer.__class__.__name__}")


if __name__ == "__main__":
    main()
