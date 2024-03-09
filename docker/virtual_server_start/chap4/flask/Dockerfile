FROM python:3.11-slim
# 該当ディレクトリがない場合作成する
WORKDIR /usr/src/app
RUN pip install flask
CMD ["flask", "run", "--host=0.0.0.0"]