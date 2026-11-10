# polars 技術調査依頼

このフォルダは skrubのTableReport に関する技術調査を記載する場所です。

## 目的

- skrubのTableReportを導入して、どのような閲覧か確認する

## 対象範囲

- skrubとfg-data-profilingの比較
- 対象データ: サンプルデータはtitanicデータを利用する
- 応用: streamlit上でのskrubの表示

## 規約

- ファイル名は内容が分かる英語のスネークケースで命名する（例: `window_functions.sql`）
- 各スクリプト・クエリにはコメントで目的・用途を簡潔に記載する
- 動作確認済みの SQL Server バージョンがあればコメントに明記する
- 関数・クラスを用いて移植性を高める