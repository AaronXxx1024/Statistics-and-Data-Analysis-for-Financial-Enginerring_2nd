"""
Problem 1 - Problem 3
The data set Stock_bond.csv contains daily volumes
and adjusted closing (AC) prices of stocks and the
S&P 500 (columns B–W) and yields on bonds (columns
X–AD) from 2-Jan-1987 to 1-Sep-2006.
"""

import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%
"""
Problem 1 
---------
1) Do the GM and Ford returns seem positively correlated? 
2) Do you notice any outlying returns? 
3) If “yes,” do outlying GM returns seem to occur with outlying Ford returns?
---------
1) Yes
2) Yes, around 2-Jan-87, 6-Nov-02, etc.
3) Yes
"""

sp500 = pd.read_csv("/Users/aaronx-mac/PycharmProjects/Learning/Github/"
                    "Statistics-and-Data-Analysis-for-Financial-Enginerring_2nd/datasets/Stock_Bond.csv", index_col=0)
print(sp500.info())

# Plot adjusted price for GM and Ford
sp500[['GM_AC', 'F_AC']].plot()
plt.show()

# Calculate daily return for GM and Ford
sp500['GMReturn'] = sp500['GM_AC'].pct_change()
sp500['FReturn'] = sp500['F_AC'].pct_change()
sp500[['GMReturn', 'FReturn']].plot(kind='scatter', x='GMReturn', y='FReturn')
plt.show()

#%%
"""
Problem 2 
---------
1) Compute the log returns for GM and plot the returns versus the log returns. 
2) How highly correlated are the two types of returns?
---------
2) 0.999541
"""

# Plot GM return and GM log return
sp500['GM_LogReturn'] = np.log(sp500['GM_AC'] / sp500['GM_AC'].shift())
sp500[['GMReturn', 'GM_LogReturn']].plot()
plt.show()

# Compute correlation of return and log return
corr = sp500[['GMReturn', 'GM_LogReturn']].corr()
print(corr)

#%%
"""
Problem 3
---------
Repeat Problem 1 with Microsoft (MSFT) and Merck (MRK).
---------

"""

# Plot adjusted price for GM and Ford
sp500[['MSFT_AC', 'MRK_AC']].plot()
plt.show()

# Calculate daily return for GM and Ford
sp500['MSFTReturn'] = sp500['MSFT_AC'].pct_change()
sp500['MRKReturn'] = sp500['MRK_AC'].pct_change()
sp500[['MSFTReturn', 'MRKReturn']].plot(kind='scatter', x='MSFTReturn', y='MRKReturn')
plt.show()

#%%
"""
Problem 4
---------
What is the probability that the value of the stock will be 
below $950,000 at the close of at least one of the next 45 
trading days? 
---------
0.9998

"""

np.random.seed(2021)

log_price = np.log(1e6)
below = [0] * int(1e5)
current_min = np.log(1e6)

for i in range(100000):
    # Create an normally distributed random variable ~ N(0.05, 0.23**2)
    r = 0.05/253 + 0.23/math.sqrt(253) * np.random.randn(45)
    log_price += sum(r)
    if log_price < current_min:
        current_min = log_price
    if current_min < np.log(950000):
        below[i] = 1

print(np.mean(below))

#%%
"""
Problem 5 
---------
What is the probability that the hedge fund will make a profit of at least $100,000? 
P(profit >= $100,00) =  0.38855
---------

Problem 6 
---------
What is the probability the hedge fund will suffer a loss? 
P(suffer a loss) =  0.58622
---------

Problem 7 
---------
What is the expected profit from this trading strategy? 
Expected Profit:  10161.415322615974
---------

Problem 8 
---------
What is the expected return? When answering this question, remember that only $50,000 was invested. 
Also, the units of return are time, e.g., one can express a return as a daily return or a weekly return. 
Therefore, one must keep track of how long the hedge fund holds its position before selling.
---------
Expected Profit:  0.0020322830645231947
"""

np.random.seed(2021)

below = [0] * int(1e5)
above = [0] * int(1e5)
middle = [0] * int(1e5)
profit = [0] * int(1e5)

for i in range(int(1e5)):
    # Create an normally distributed random variable ~ N(0.05, 0.23**2)
    r = 0.05/253 + 0.23/math.sqrt(253) * np.random.randn(100)
    current_ret = 0
    profit[i] = np.exp(np.log(1e6) + sum(r)) - 1e6
    for ret in r:
        current_ret += ret
        log_price = np.log(1e6) + current_ret
        if log_price <= np.log(950000):
            below[i] = 1
            profit[i] = -50000
            break
        elif log_price >= np.log(1100000):
            above[i] = 1
            profit[i] = 100000
            break
        middle[i] = 1

expected_return = (np.array(profit) / 50000.0) / 100

print("P(profit >= $100,00) = " , np.mean(above))
print("P(suffer a loss) = " , np.mean(below))
print("P(price between $950,000 & $1,100,000) =  ", np.mean(middle))
print("Expected Profit: ", np.mean(profit))
print("Expected Profit: ", np.mean(expected_return))

#%%
"""
Problem 9 
---------
In this simulation, what are the mean and standard deviation of the log-returns for 1 year? 
mean: 0.001206613351199386
std: 0.012675694201796584
---------

Problem 10 
---------
Discuss how the price series appear to have momentum. Is the appearance of momentum real or an illusion? 
illusion
---------

Problem 11 
---------
Explain what the code c(120,120*exp(cumsum(logr))) does?
create a price series movement started from 120, total length is 254.
---------

"""

np.random.seed(2021)

price = []
logr = 0.05/253 + 0.2/math.sqrt(253) * np.random.randn(253)
current_r = 0
for r in logr:
    current_r += r
    price.append(np.exp(current_r) * 120)

pd.Series(price).plot()
plt.show()

print(np.mean(logr))
print(np.std(logr))


