# MLflow サンプル (Titanic)

Titanicデータセットを題材に、**MLflowの一般的な使い方**(experiment管理・パラメータ/メトリクスのロギング・モデルの保存)を示すサンプルコードです。
モデル精度そのものは目的ではなく、**Windows/Linuxが混在する環境でも問題なく動作する構成**であることを主眼としています。

詳しい設計方針・背景は [CLAUDE.md](./CLAUDE.md) を参照してください。

## この構成のポイント

- **backend store**: SQLite 1ファイル(`mlflow_env/mlflow.db`)。追加ミドルウェア不要。
- **artifact store**: **proxied artifact access**。クライアントはartifactの実ファイルに
  直接アクセスせず、必ず HTTP 経由で mlflow server に対してアップロード/ダウンロードをリクエストする。
  そのため新規experimentのartifact URIは `mlflow-artifacts:/...` という抽象スキームになり、
  `file:///C:/Users/...` のようなOS依存パスがクライアントに一切露出しない。
- クライアント側は `boto3` や `pyodbc` 等の追加ドライバが**不要**。環境変数
  `MLFLOW_TRACKING_URI` を設定するだけで、Windows/Linuxどちらのクライアントからも同じように使える。

## ディレクトリ構成

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── start_mlflow_server.bat        # Windows側でmlflow serverを起動するバッチ
├── data/
│   └── titanic.csv                # 初回実行時にseabornから自動取得・キャッシュされる
├── src/
│   ├── data_loader.py             # データ取得・前処理
│   ├── mlflow_utils.py            # mlflow関連処理の薄いラッパー
│   └── train.py                   # 学習〜ロギングの一連のサンプルフロー
└── mlflow_env/                    # サーバー側の実データ置き場(.gitignore対象)
    ├── mlflow.db                  # backend store (sqlite)
    └── mlartifacts/               # artifactsの実体(proxied経由でのみアクセスされる)
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行手順

### 1. サーバーの起動(サーバー役のWindows機で1回だけ)

```bat
start_mlflow_server.bat
```

- `host=0.0.0.0`, `port=5000` で起動します(他OSクライアントからの接続を受け付けるため)。
- `mlflow_env/` がなければ自動的に作成されます。
- **このサーバープロセスは常に1つだけ**起動してください(SQLiteは複数プロセスからの
  同時直接書き込みに弱いため)。

### 2. クライアント側の環境変数設定

接続設定は環境変数 `MLFLOW_TRACKING_URI` のみで行います。コード内にIPアドレスや
ポートをハードコードすることはありません。

Windows (コマンドプロンプト):
```bat
set MLFLOW_TRACKING_URI=http://<サーバーIP>:5000
```

Linux / macOS (bash):
```bash
export MLFLOW_TRACKING_URI=http://<サーバーIP>:5000
```

未設定の場合は `http://localhost:5000` にフォールバックし、警告ログが出力されます
(`src/mlflow_utils.py` の `get_tracking_uri()` を参照)。

### 3. 学習スクリプトの実行

```bash
python src/train.py
```

Titanicの生存予測を `RandomForestClassifier` で学習し、以下をMLflowへ記録します。

| 種別 | 内容 |
|---|---|
| パラメータ | モデル名、`n_estimators`, `max_depth`, `random_state` |
| メトリクス | accuracy, precision, recall, f1 |
| モデル本体 | `mlflow.sklearn.log_model` |
| artifact | 混同行列画像(`confusion_matrix.png`) |

### 4. 動作確認

MLflow UI (`http://<サーバーIP>:5000`) を開き、以下を確認してください。

- experiment `titanic-survival-classification` に今回のrunが記録されていること。
- runの詳細画面で、**artifact_locationが `mlflow-artifacts:/...` になっていること**
  (`file://...` になっていた場合はサーバー起動設定のミスです。
  `start_mlflow_server.bat` の `--artifacts-destination` 指定と、
  `--default-artifact-root` を指定していないことを確認してください)。
- runのartifactsに `model/` と `confusion_matrix.png` が保存されていること。

さらに、**Linux側・Windows側の両方**から同じ `MLFLOW_TRACKING_URI` を設定した状態で、
同一runのartifactを取得できることを確認してください。

```python
import mlflow

mlflow.artifacts.download_artifacts(
    run_id="<UIで確認したrun_id>",
    artifact_path="model",
    dst_path="./downloaded_model",
)
```

OS依存のパス不一致でエラーになることなく、Windows/Linuxどちらからもダウンロードできれば
proxied artifact accessが正しく機能しています。

## 依存パッケージ

```
mlflow==3.8.1
scikit-learn
pandas
seaborn
matplotlib
```

MinIO関連(`boto3`)、SQL Server関連(`pyodbc`)など、今回の構成では不要な依存は
追加していません。
