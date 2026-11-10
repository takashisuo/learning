/*
コメントだよ
*/
declare @ADDRESS VARCHAR(255);
set @ADDRESS = '東京都千代田区丸の内1-1-1';

with
    BASE as (
        select
            test
            , test2
        from EMPLOYEES
        where ISACTIVE = 1
    )

    , table1 as (
        select
            test
            , test2
        from EMPLOYEES
        where
            1 = 1
            and test = 1
            and test2 = 2
    )

    , table2 as (
        select
            test
            , test2
        from EMPLOYEES
        where 1 = 1
    )
    , table3 as (
        select
            MAX(test) as test
        from EMPLOYEES
    )

select
    -- testtesttest
    BASE.test
    , BASE.年収金額 as test1
    , BASE.年収金額 as test2
    , table1.test2 as test22
from BASE
left outer join
    table1
    on BASE.test = table1.test
