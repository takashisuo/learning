FROM python:3.11-slim

# 作業ディレクトリを変更
COPY requirements.txt ${PWD}

# pipアップデート
RUN pip install -U pip

# pythonパッケージをインストール
RUN pip install -r requirements.txt

# 一時変数の定義
ARG wdir

# 作業ディレクトリの変更
WORKDIR $wdir