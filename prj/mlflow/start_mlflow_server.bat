@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  MLflow tracking server 起動バッチ (Windows / サーバー役)
REM
REM  - backend store : SQLite (mlflow_env\mlflow.db)
REM  - artifact store: proxied access (--artifacts-destination)
REM    * --default-artifact-root は指定しないこと
REM      (指定すると生の file:// パスがクライアントに露出してしまう)
REM  - このサーバープロセスは常に1つだけ起動すること
REM ============================================================

REM 必要に応じて仮想環境を有効化する(利用環境に合わせてコメントアウトを外す)
REM call conda activate mlflow-env
REM call "%~dp0.venv\Scripts\activate.bat"

REM %~dp0 はこのバッチファイル自身のディレクトリ(末尾に \ 付き)。
REM スペースを含むパスに対応するため、変数展開時は必ずダブルクォートで囲む。
set "MLFLOW_HOME=%~dp0mlflow_env"
set "DB_PATH=%MLFLOW_HOME%\mlflow.db"
set "ARTIFACT_ROOT=%MLFLOW_HOME%\mlartifacts"

if not exist "%MLFLOW_HOME%" (
    echo [INFO] %MLFLOW_HOME% が存在しないため作成します。
    mkdir "%MLFLOW_HOME%"
    if errorlevel 1 (
        echo [ERROR] %MLFLOW_HOME% の作成に失敗しました。
        goto :error
    )
)

if not exist "%ARTIFACT_ROOT%" (
    mkdir "%ARTIFACT_ROOT%"
)

where mlflow >nul 2>nul
if errorlevel 1 (
    echo [ERROR] mlflow コマンドが見つかりません。
    echo         仮想環境の有効化、または requirements.txt のインストールを確認してください。
    goto :error
)

REM backend-store-uri / artifacts-destination はURI形式にするためスラッシュに変換する。
REM (Windowsのバックスラッシュ区切りのままだと sqlite:/// 等のURIとして解釈できないため)
set "DB_PATH_URI=%DB_PATH:\=/%"
set "ARTIFACT_ROOT_URI=%ARTIFACT_ROOT:\=/%"

echo [INFO] MLFLOW_HOME          = %MLFLOW_HOME%
echo [INFO] backend-store-uri    = sqlite:///%DB_PATH_URI%
echo [INFO] artifacts-destination = file:///%ARTIFACT_ROOT_URI%
echo [INFO] mlflow server を起動します (host=0.0.0.0 port=5000) ...
echo.

mlflow server ^
  --host 0.0.0.0 ^
  --port 5000 ^
  --backend-store-uri "sqlite:///%DB_PATH_URI%" ^
  --artifacts-destination "file:///%ARTIFACT_ROOT_URI%"

if errorlevel 1 (
    echo [ERROR] mlflow server の起動に失敗しました。
    goto :error
)

goto :end

:error
echo.
echo [ERROR] mlflow server を起動できませんでした。上記のログを確認してください。
pause
exit /b 1

:end
pause
