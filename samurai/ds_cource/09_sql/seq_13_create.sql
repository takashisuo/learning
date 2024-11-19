# 部署テーブルを作成
CREATE TABLE IF NOT EXISTS departments (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  dep_name VARCHAR(60) NOT NULL
);

# 部署テーブルにレコードを追加
INSERT INTO departments (id, dep_name) VALUES
  (1001, '企画部'),
  (1002, '技術部'),
  (1003, '営業部');

# 社員テーブルを作成
CREATE TABLE IF NOT EXISTS employees (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  emp_name VARCHAR(60) NOT NULL,
  dep_id INT NOT NULL,
  FOREIGN KEY (dep_id) REFERENCES departments(id)
);

# 社員テーブルにレコードを追加
INSERT INTO employees (emp_name, dep_id) VALUES
  ('侍健太',     1002),
  ('刀沢彩香',   1001),
  ('戦国広志',   1004),
  ('武士山美咲', 1002);
