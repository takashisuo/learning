# CLAUDE.md

このファイルは、本リポジトリで Claude Code が作業する際に従うべきプロジェクト方針をまとめたものです。

## プロジェクトの目的

Titanicデータセットを題材に、**MLflowの一般的な使い方(experiment管理・パラメータ/メトリクスのロギング・モデルの保存)を抽象化したサンプルコード**を作成する。
モデル精度そのものは目的ではない。**Windows/Linuxが混在する環境でも問題なく動作する構成**であることを最優先する。

---

## 背景(なぜこの構成にするか)

これまでの検討で、以下の問題と解決方針が決まっている。Claude Codeはこの前提を崩す実装をしないこと。

### 問題
MLflowのデフォルト構成(ローカルファイルパスを直接artifact保存先にする方式)では、artifactの場所が `file:///C:/Users/xxx/...` や `file:///home/xxx/...` のようなOS依存の絶対パスとしてDBに記録される。Windows側で記録したrunをLinux側から読もうとすると、パス不一致でアクセスできない。

### 解決方針
1. **backend store**: SQLite(`sqlite:///.../mlflow.db`)を使用。1ファイルで完結し、追加ミドルウェア不要。
2. **artifact store**: **proxied artifact access** を使用する。
   - `mlflow server` 起動時に `--artifacts-destination` を指定し、`--default-artifact-root` / 旧来の生の `file://` 直接指定は使わない。
   - これにより、新規experimentのartifact URIは `mlflow-artifacts:/...` という抽象スキームになる。
   - クライアント(Windows/Linuxいずれも)は、artifactの実ファイルに直接アクセスするのではなく、**必ずHTTP経由でmlflow serverに対してアップロード/ダウンロードをリクエストする**。
   - 結果として、クライアント側にOS依存のファイルパスが一切露出しない。
3. **mlflow serverプロセスは常に1つだけ**稼働させる。SQLiteは複数プロセスからの同時直接書き込みに弱いため、必ずこのサーバー経由でのみメタデータ・artifactにアクセスさせること。
4. クライアントコードは `MLFLOW_TRACKING_URI`(例: `http://<サーバーIP>:5000`)を通してのみサーバーに接続し、**ローカルファイルパスやドライブレター(`Z:\` など)をコード中にハードコードしない**。

### 今回のスコープ外(採用しない選択肢)
- MinIO / S3互換オブジェクトストレージ: 将来的なクラウド移行やスケールが必要になった場合の選択肢として検討済みだが、現状(数名・低頻度アクセス)ではオーバースペックのため不採用。
- SQL Server backend: 既存のSQL Server接続実績はあるが、数名規模の利用にはSQLiteで十分なため不採用。
- Model Registry(モデルのバージョン管理・Staging/Production等のステージ管理): 今回は使わない想定。
- 上記に関連する依存パッケージ(`boto3`, `pyodbc` 等)は本プロジェクトに追加しないこと。

### MLflowのバージョンについて(重要)

本プロジェクトは **mlflow==3.8.1** を前提とする(`requirements.txt`にバージョンを固定すること)。

背景として、MLflowは filesystem backend(`backend-store-uri`に`./mlruns`等の生のファイルパスを直接使う方式)を段階的に廃止している。

| バージョン | 内容 |
|---|---|
| 3.6 | filesystem backendの使用に対して非推奨警告(FutureWarning)を追加 |
| 3.7 | 何も指定しない場合のデフォルトが `./mlruns` から `sqlite:///mlflow.db` に変更 |
| 3.13 | filesystem backendの使用が警告ではなく**例外(起動エラー)**になった(`MLFLOW_ALLOW_FILE_STORE=true`を明示しない限り使用不可) |

**本プロジェクトの構成はこの廃止の影響を受けない。** 理由は、`backend-store-uri`に生のfile://パスを使わず**最初からsqliteを指定している**ため。なお、廃止対象はあくまで「backend store(メタデータ)」としてのfilesystem利用であり、「artifact store」としてのfile://指定(`--artifacts-destination file:///...`)は将来のバージョンでも引き続きサポートされる、別物の機能である。この区別を混同した実装をしないこと。

---

## ディレクトリ構成(この構成に従うこと)

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── start_mlflow_server.bat        # Windows側でmlflow serverを起動するバッチ
├── data/
│   └── titanic.csv                # なければ実行時に自動取得するロジックを用意する
├── src/
│   ├── data_loader.py             # データ取得・前処理
│   ├── mlflow_utils.py            # mlflow関連処理の薄いラッパー(抽象化の中心)
│   └── train.py                   # 学習〜ロギングの一連のサンプルフロー
└── mlflow_env/                    # サーバー側の実データ置き場(gitignore対象)
    ├── mlflow.db                  # backend store (sqlite)
    └── mlartifacts/               # artifactsの実体(proxied経由でのみアクセスされる)
```

`mlflow_env/` はサーバー機のローカル、または共有ドライブ上に置く想定。**このディレクトリの実体パスをクライアントコードが直接参照することは絶対にない**設計にすること(参照したくなった場合は設計ミスなので `mlflow_utils.py` を見直す)。

---

## 実装方針

### 1. データ (`src/data_loader.py`)
- Titanicデータセットは `seaborn.load_dataset("titanic")` での取得を第一候補とする。オフライン環境等で使えない場合のフォールバックとして `data/titanic.csv` を読み込む形にする。
- パス操作は必ず `pathlib.Path` を使用し、`os.sep` やドライブレターに依存する処理を書かないこと。
- 相対パス(スクリプトファイルからの相対位置)で解決し、絶対パスの決め打ちを禁止する。

### 2. MLflow処理の抽象化 (`src/mlflow_utils.py`)
以下のような薄いラッパー関数/クラス群として実装し、`train.py` 側がMLflowのAPI詳細を意識しなくてよい形にする。

- `get_tracking_uri()`: 環境変数 `MLFLOW_TRACKING_URI` から取得する。未設定時はデフォルト値(例: `http://localhost:5000`)にフォールバックし、警告ログを出す。
- `setup_experiment(experiment_name: str)`: tracking_uriの設定 + `mlflow.set_experiment()` をまとめて行う。
- `run_experiment(experiment_name, params, metrics, model, model_flavor, artifacts=None)`: `with mlflow.start_run():` の中で `log_param` / `log_metric` / `log_model` / (必要なら)`log_artifact` を行う一連の処理をまとめる。
- 具体的なモデルの種類(scikit-learnのRandomForest等)に依存しすぎないよう、`model_flavor`(例: `"sklearn"`)を引数で受け取り、内部で対応する `mlflow.<flavor>.log_model` を呼び分ける程度の抽象化にとどめる(過度に汎用化しすぎない)。

### 3. 学習サンプル (`src/train.py`)
- Titanicの生存予測を題材に、シンプルな分類モデル(例: `RandomForestClassifier`)を1つ用意すれば十分。複数モデルの比較などは行わない(サンプルの目的はMLflowの使い方を示すことであり、モデル精度の追求ではない)。
- ログする内容の例:
  - パラメータ: モデル名、主要なハイパーパラメータ(`n_estimators` 等)
  - メトリクス: accuracy, precision, recall, f1
  - モデル本体: `mlflow.sklearn.log_model`
  - 任意のartifact例: 混同行列の画像など(proxied access経由で保存されることの確認も兼ねる)

### 4. 起動確認用の記述
`train.py` 実行後、以下を確認できるようREADMEまたはコード内コメントに明記すること。
- 記録されたrunのartifact_locationが `mlflow-artifacts:/...` になっていること(`file://...` になっていたら設定ミス)
- Linux側・Windows側の両方から同じrunのartifactを `mlflow.artifacts.download_artifacts()` 等で取得できること

---

## mlflow server起動バッチ (`start_mlflow_server.bat`)

Windows機(サーバー役)で実行する前提のバッチファイル。以下の要件を満たすこと。

- `host`: `0.0.0.0`(他OSクライアントからの接続を受け付けるため)
- `port`: `5000`
- `backend-store-uri`: `sqlite:///` + `mlflow_env/mlflow.db` への絶対パス
- `artifacts-destination`: `file:///` + `mlflow_env/mlartifacts` への絶対パス
- `--default-artifact-root` は**指定しないこと**(proxied modeを有効にするため。指定すると生のfile://パスがクライアントに露出し、今回解決したい問題が再発する)
- 実行前に、必要であれば仮想環境(venv/conda)を有効化する処理を含める
- 想定利用者は数名・低頻度アクセスのため、`--workers` 等の高度なチューニングは不要

### 期待するコマンドの骨格(参考)

```bat
@echo off
REM 必要に応じて仮想環境を有効化
REM call conda activate mlflow-env

set MLFLOW_HOME=%~dp0mlflow_env

mlflow server ^
  --host 0.0.0.0 ^
  --port 5000 ^
  --backend-store-uri sqlite:///%MLFLOW_HOME%/mlflow.db ^
  --artifacts-destination file:///%MLFLOW_HOME%/mlartifacts

pause
```

上記はあくまで骨格であり、Claude Codeが実装する際はパスの組み立て(スペースを含むパス対応、`%~dp0` の扱いなど)やエラーハンドリングを適切に行うこと。

---

## クライアント側の利用ルール(Windows/Linux共通)

- 接続設定は環境変数 `MLFLOW_TRACKING_URI=http://<サーバーIP>:5000` のみで行う。コード内にIPアドレスやポートをハードコードしない(ドキュメント中のサンプルを除く)。
- クライアント側に `boto3` や `pyodbc` 等、artifact/backend store用の追加ドライバは一切不要である(proxied access + SQLiteのため)。これが今回の構成の利点であることをREADMEに明記する。

---

## 依存パッケージ (`requirements.txt`)

最小限に留めること。目安は以下の通り(バージョン固定は任意)。samurai仮想環境で問題なければそのまま利用すること。

```
mlflow==3.8.1
scikit-learn
pandas
seaborn
```

MinIO関連(`boto3`)、SQL Server関連(`pyodbc`)は追加しないこと。

---

## Claude Codeへの実装時の注意

- 新規ファイル作成時は上記ディレクトリ構成に従うこと。
- `mlflow_env/` は `.gitignore` に追加し、リポジトリにコミットしないこと。
- サンプルコードは「MLflowの一般的な使い方を示す」ことが目的なので、過度に複雑な抽象化(汎用フレームワーク化)は避け、読んでそのまま理解できる薄いラッパーに留めること。
- 実装完了後、`start_mlflow_server.bat` 実行 → `python src/train.py` 実行 → MLflow UI (`http://<サーバーIP>:5000`) でrunとartifactが確認できる、という一連の動作確認手順をREADMEに記載すること。
