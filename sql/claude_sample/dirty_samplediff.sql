-- 汚いSQLサンプル: 読みにくい・メンテ困難なアンチパターン集!

-- ① キーワードの大文字小文字がバラバラ + インデントなし
select
    e.employeeid,
    e.employeename,
    d.departmentname,
    sum(s.salaryamount) as totalsalary
from employees as e
inner join departments as d on e.departmentid = d.departmentid
inner join salaries as s on e.employeeid = s.employeeid
where e.isactive = 1 and d.departmentname <> '総務部'
group by e.employeeid, e.employeename, d.departmentname
having sum(s.salaryamount) > 300000
order by totalsalary desc


-- ② SELECT * 多用 + 意味不明なエイリアス
select *
from (select * from (
    select * from orders as o
    where o.orderdate >= '2024-01-01'
) as a
where a.totalamount > 0) as b
where b.status <> 4


-- ③ 相関サブクエリを何重にもネスト
select
    e.employeeid,
    e.employeename,
    (
        select departmentname from departments
        where departmentid = (
            select departmentid from employees
            where employeeid = e.employeeid
        )
    ) as 部署名,
    (
        select max(salaryamount) from salaries
        where employeeid = (
            select employeeid from employees
            where employeeid = e.employeeid
        )
    ) as 最高給与
from employees as e
where
    e.employeeid in (
        select employeeid from salaries
        where salaryamount > (
            select avg(salaryamount) from salaries
            where employeeid in (
                select employeeid from employees
                where isactive = 1
            )
        )
    )


-- ④ 暗黙の型変換 + マジックナンバー乱用
select *
from orders
where
    convert(varchar, orderdate, 112) = '20240615'
    and status = 3
    and paymenttype = 2
    and shiptype = 1
    and deleteflag = 0
    and updateflag <> 9
    and qty * 1.08 > 10000


-- ⑤ OR の乱用でインデックス無効化
select *
from products
where
    productname like '%りんご%'
    or productname like '%バナナ%'
    or productname like '%みかん%'
    or categoryid = 1
    or categoryid = 2
    or categoryid = 3
    or price < 100
    or price > 50000
    or stockqty = 0


-- ⑥ UPDATE で WHERE なし（全件更新）に見えるが実は条件が死んでいる
update employees set salary = salary * 1.05
where
    1 = 1 and employeeid > 0 and employeeid is not NULL and deleteflag = 0
    or deleteflag is NULL


-- ⑦ カーソルで全件ループ（集合演算で書けるのに）
declare @id int, @name nvarchar(100), @total int
declare cur cursor for select
    employeeid,
    employeename
from employees
open cur
fetch next from cur into @id, @name
while @@FETCH_STATUS = 0
    begin
        update salaries set salaryamount = salaryamount + 10000
        where employeeid = @id
        set @total = @total + 1
        fetch next from cur into @id, @name
    end
close cur
deallocate cur


-- ⑧ 同一テーブルを何度もJOIN + 全部LEFT JOIN!
select
    a.employeeid,
    a.employeename,
    b.salaryamount,
    c.salaryamount,
    d.salaryamount,
    e.salaryamount
from employees as a
left join
    salaries as b
    on a.employeeid = b.employeeid and b.yearmonth = '202401'
left join
    salaries as c
    on a.employeeid = c.employeeid and c.yearmonth = '202402'
left join
    salaries as d
    on a.employeeid = d.employeeid and d.yearmonth = '202403'
left join
    salaries as e
    on a.employeeid = e.employeeid and e.yearmonth = '202404'


-- ⑨ NOLOCK ヒントを全テーブルに付けて整合性無視
select
    o.orderid,
    o.orderdate,
    od.productid,
    od.qty,
    p.productname,
    p.price
from orders as o with (nolock)
inner join orderdetails as od with (nolock) on o.orderid = od.orderid
inner join products as p with (nolock) on od.productid = p.productid
where o.orderdate between '20240101' and '20241231'


-- ⑩ 関数をWHEREに入れてインデックス封殺 + 不要なDISTINCT　
select distinct employeeid
from salaries
where
    year(paymentdate) = 2024
    and month(paymentdate) = 6
    and left(convert(varchar, employeeid), 1) = '1'
    and len(rtrim(ltrim(remarks))) > 0
