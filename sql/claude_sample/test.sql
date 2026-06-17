with BASE as (
    select
        test
        , test2
    from EMPLOYEES
    where ISACTIVE = 1
)

select
    test
    , test2
from
    BASE
