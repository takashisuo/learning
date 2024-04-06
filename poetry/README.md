1. pip install poetry (condaで作成した仮想環境上で実行する)
2. 作成したい上位フォルダからpoetry new prj名をおこなう。
    例: poetryというフォルダからpoetry new poetry-demoをおこなうと、poetryの中にpoetry-demoフォルダが生成される。
3. poetry add requests でpip install requestsと同じ効果がある
4. 実行は poetry run python main.py のように実行する。コマンドが長いのが面倒だが..その場合はpoetry shellで仮想環境に入ること。
5. poetry.lockはgit管理すること。そうすればバージョンの差異が発生しない。
6. exitでshellから抜ける
7. pythonのパスなどはpoetry ewv info でわかる。Executableのパス。
8. VSCode
    VSCode上で設定を開き、「Python: Venv Path」に先ほど確認したパス (~/Library/Caches/pypooetry/virtualenvs) を入力します。
    それか https://zenn.dev/alivelimb/articles/20220501-python-env ではpoetryの実行環境をプロジェクト直下に作成することで選択せず対応できる。

