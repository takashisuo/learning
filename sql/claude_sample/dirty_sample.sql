-- 汚いSQLサンプル: 読みにくい・メンテ困難なアンチパターン集!

-- ① キーワードの大文字小文字がバラバラ + インデントなし
select e.EmployeeID,e.EmployeeName,d.DepartmentName,sum(s.SalaryAmount) as TotalSalary from Employees e join Departments d on e.DepartmentID=d.DepartmentID join Salaries s on s.EmployeeID=e.EmployeeID where e.IsActive=1 and d.DepartmentName<>'総務部' group by e.EmployeeID,e.EmployeeName,d.DepartmentName having sum(s.SalaryAmount)>300000 order by TotalSalary desc


-- ② SELECT * 多用 + 意味不明なエイリアス
SELECT * FROM (SELECT * FROM (SELECT * FROM Orders o WHERE o.OrderDate >= '2024-01-01') AS a WHERE a.TotalAmount > 0) AS b WHERE b.Status != 4


-- ③ 相関サブクエリを何重にもネスト
SELECT EmployeeID, EmployeeName, (SELECT DepartmentName FROM Departments WHERE DepartmentID = (SELECT DepartmentID FROM Employees WHERE EmployeeID = e.EmployeeID)) AS 部署名, (SELECT MAX(SalaryAmount) FROM Salaries WHERE EmployeeID = (SELECT EmployeeID FROM Employees WHERE EmployeeID = e.EmployeeID)) AS 最高給与 FROM Employees e WHERE e.EmployeeID IN (SELECT EmployeeID FROM Salaries WHERE SalaryAmount > (SELECT AVG(SalaryAmount) FROM Salaries WHERE EmployeeID IN (SELECT EmployeeID FROM Employees WHERE IsActive = 1)))


-- ④ 暗黙の型変換 + マジックナンバー乱用
select * from Orders where CONVERT(varchar,OrderDate,112) = '20240615' and Status=3 and PaymentType=2 and ShipType=1 and DeleteFlag=0 and UpdateFlag <> 9 and Qty * 1.08 > 10000


-- ⑤ OR の乱用でインデックス無効化
SELECT * FROM Products WHERE ProductName LIKE '%りんご%' OR ProductName LIKE '%バナナ%' OR ProductName LIKE '%みかん%' OR CategoryID = 1 OR CategoryID = 2 OR CategoryID = 3 OR Price < 100 OR Price > 50000 OR StockQty = 0


-- ⑥ UPDATE で WHERE なし（全件更新）に見えるが実は条件が死んでいる
UPDATE Employees SET Salary = Salary * 1.05 WHERE 1=1 AND EmployeeID > 0 AND EmployeeID IS NOT NULL AND DeleteFlag = 0 OR DeleteFlag IS NULL


-- ⑦ カーソルで全件ループ（集合演算で書けるのに）
DECLARE @id INT, @name NVARCHAR(100), @total INT
DECLARE cur CURSOR FOR SELECT EmployeeID, EmployeeName FROM Employees
OPEN cur
FETCH NEXT FROM cur INTO @id, @name
WHILE @@FETCH_STATUS = 0
BEGIN
UPDATE Salaries SET SalaryAmount = SalaryAmount + 10000 WHERE EmployeeID = @id
SET @total = @total + 1
FETCH NEXT FROM cur INTO @id, @name
END
CLOSE cur
DEALLOCATE cur


-- ⑧ 同一テーブルを何度もJOIN + 全部LEFT JOIN!
SELECT a.EmployeeID,a.EmployeeName,b.SalaryAmount,c.SalaryAmount,d.SalaryAmount,e.SalaryAmount FROM Employees a LEFT JOIN Salaries b ON a.EmployeeID=b.EmployeeID AND b.YearMonth='202401' LEFT JOIN Salaries c ON a.EmployeeID=c.EmployeeID AND c.YearMonth='202402' LEFT JOIN Salaries d ON a.EmployeeID=d.EmployeeID AND d.YearMonth='202403' LEFT JOIN Salaries e ON a.EmployeeID=e.EmployeeID AND e.YearMonth='202404'


-- ⑨ NOLOCK ヒントを全テーブルに付けて整合性無視
SELECT o.OrderID,o.OrderDate,od.ProductID,od.Qty,p.ProductName,p.Price FROM Orders o WITH(NOLOCK) JOIN OrderDetails od WITH(NOLOCK) ON o.OrderID=od.OrderID JOIN Products p WITH(NOLOCK) ON od.ProductID=p.ProductID WHERE o.OrderDate BETWEEN '20240101' AND '20241231'


-- ⑩ 関数をWHEREに入れてインデックス封殺 + 不要なDISTINCT
SELECT DISTINCT EmployeeID FROM Salaries WHERE YEAR(PaymentDate)=2024 AND MONTH(PaymentDate)=6 AND LEFT(CONVERT(varchar,EmployeeID),1)='1' AND LEN(RTRIM(LTRIM(Remarks)))>0
