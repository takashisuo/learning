from scipy import stats
import numpy as np
"""
kstest: コルモゴロフ-スミルノフ検定
"""

def call_kstest1():
    data: np.ndarray = stats.norm.rvs(size=500)
    # [1]: 
    pv = stats.kstest(data, "norm")[1]
    print(f'p-value: {pv}')
    # p値=正しさ, 大きいほうが帰無仮説を棄却できない。
    if pv > 0.05:
        print("帰無仮説を棄却できない。すなわち正規分布でないとは言えない。")
    else:
        print("帰無仮説を棄却する。すなわち分布に差がある。")
        
def call_kstest2():
    mu = 5
    sigma2 = 3
    data: np.ndarray = stats.norm.rvs(size=500) * sigma2 + mu
    # [1]: 
    pv = stats.kstest(data, stats.norm(loc=mu, scale=sigma2).cdf)[1]
    print(f'p-value: {pv}')
    # p値=正しさ, 大きいほうが帰無仮説を棄却できない。
    if pv > 0.05:
        print("帰無仮説を棄却できない。すなわち正規分布でないとは言えない。")
    else:
        print("帰無仮説を棄却する。すなわち分布に差がある。")

def call_kstest3():
    mu = 5
    sigma2 = 3
    data: np.ndarray = stats.norm.rvs(size=500) * sigma2 + mu
    pv = stats.kstest(data, stats.norm(loc=mu + 1, scale=sigma2).cdf)[1]
    print('p-value:' + str(pv))
    if pv > 0.05:
        print("帰無仮説を棄却できない。すなわち正規分布でないとは言えない。")
    else:
        print("帰無仮説を棄却する。すなわち分布に差がある。")

if __name__ == "__main__":
    call_kstest1()
    call_kstest2()
    call_kstest3()