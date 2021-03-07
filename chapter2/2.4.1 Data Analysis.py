"""
Part 1: Problem 1 - Problem 17
Part 2: Exercise 1 - Exercise 11
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


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

#%%
"""
Problem 12
---------- 
Compute the returns and log returns and plot them against each other. 
As discussed in Sect. 2.1.3, does it seem reasonable that the two types of daily returns are approximately equal? 
---------- 
Yes.

Problem 13 
----------
Compute the mean and standard deviation for both the returns and the log returns. 
Comment on the similarities and differences you perceive in the first two moments of each random variable. 
Does it seem reasonable that they are the same? 
---------- 
Mean of return:  0.0005027479017916817
Mean of log return:  0.00046305531799950865
Std of return:  0.00890031928901851
Std of log return:  0.008901467457248666
Pretty similar but not exactly same.

Problem 14
----------
1) Perform a t-test to compare the means of the returns and the log returns. 
   Comment on your findings. Do you reject the null hypothesis that they are 
   the same mean at 5 % significance? Or do you accept it? 
   [Hint: Should you be using an independent samples t-test or a paired-samples t-test?] 
2) What are the assumptions behind the t-test? Do you think that they are met in this example? 
3) If the assumptions made by the t-test are not met, how would this affect your interpretation 
   of the results of the test? 
---------- 
1) Paired t-test result:
   Ttest_relResult(statistic=15.865603933357201, pvalue=1.642607974463449e-51)
   Since p-value is very small, we reject the null hypothesis of equal mean at 5% significance.
2) The difference series calculated from return & log_return is independent and normally distributed
   with a constant variance.
3) From solution manual:

Problem 15
----------
After looking at return and log return data for McDonald’s, are you satisfied that for small values, 
log returns and returns are interchangeable?
---------- 
Yes.

Problem 16
----------
Assume that McDonald’s log returns are normally distributed with mean and standard deviation 
equal to their estimates and that you have been made the following proposition by a friend: 
If at any point within the next 20 trading days, the price of McDonald’s falls below 85 dollars, 
you will be paid $100, but if it does not, you have to pay him $1. The current price of McDonald’s 
is at the end of the sample data, $93.07. 
Are you willing to make the bet? 
(Use 10,000 iterations in your simulation and use the command set.seed(2015) to ensure your results 
are the same as the answer key)
----------
I got an negative expected payoff, which is different with the solution....

"""
# read data
mcd = pd.read_csv('./datasets/MCD_PriceDaily.csv')

# create return series
mcd['return'] = mcd['Adj Close'].pct_change()
mcd['log_return'] = np.log(mcd['Adj Close'] / mcd['Adj Close'].shift())

# plot return
plt.scatter(x=mcd['return'], y=mcd['log_return'])
plt.show()

# mean and std for each return series
print("Mean of return: ", mcd['return'].mean())
print("Mean of log return: ", mcd['log_return'].mean())
print("Std of return: ", mcd['return'].std())
print("Std of log return: ", mcd['log_return'].std())

# paired t-test
print(stats.ttest_rel(a=mcd['return'][1:], b=mcd['log_return'][1:]))
# plot distribution of log return
mcd['log_return'].hist(bins=60)
plt.show()

# P16 & 17
np.random.seed(2015)

log_mean = mcd['log_return'].mean()
log_std = mcd['log_return'].std()
win = [-1] * 10000
p0 = list(mcd['Adj Close'])[-1]
log_p0 = np.log(p0)

for i in range(10000):
    logr = log_mean + log_std * np.random.randn(20)
    r0 = 0
    for r in logr:
        r0 += r
        if p0 * np.exp(r0) < 84.5:
            win[i] = 25
            break

print("Expected payoff: ", np.mean(win))

#%%
"""
Exercises 1 - 7
1. Suppose that the daily log returns on a stock are independent and normally distributed 
with mean 0.001 and standard deviation 0.015. Suppose you buy $1,000 worth of this stock.
(a) What is the probability that after one trading day your investment is worth less than $990? 
    (Note: The R function pnorm() will compute a normal CDF, so, for example, pnorm(0.3, mean = 0.1, 
    sd = 0.2) is the normal CDF with mean 0.1 and standard deviation 0.2 evaluated at 0.3.) 
    0.23065573155475771
(b) What is the probability that after five trading days your investment is worth less than $990
    0.3268188763247845
    
2. The yearly log returns on a stock are normally distributed with mean 0.1 and standard deviation 0.2. 
The stock is selling at $100 today. What is the probability that 1 year from now it is selling at $110 
or more?
0.5093539805793366

3. The yearly log returns on a stock are normally distributed with mean 0.08 and standard deviation 0.15. 
The stock is selling at $80 today. What is the probability that 2 years from now it is selling at $90 or more?
0.5788735865041656

4. Suppose the prices of a stock at times 1, 2, and 3 are P1= 95, P2= 103, and P3= 98. Find r3(2).
0.03157894736842115

5. The prices and dividends of a stock are given in the table below.
(a) What is R2? 
0.04230769230769238
(b) What is R4(3)? 
0.147958796968231
(c) What is r3?
-0.014925650216675593

7.a
N(0.24, 1.88)
7.b
0.900361112462998
7.c
0.47
7.d
N(0.72, 0.94)
"""

# E1
# Find the P(log(r1)<=log(990/1000))
# 1.a
stats.norm.cdf(np.log(990/1000), loc=0.001, scale=0.015)
# 1.b
stats.norm.cdf(np.log(990/1000), loc=0.005, scale=0.015*np.sqrt(5))
# 2
1 - stats.norm.cdf(np.log(110/100), loc=0.1, scale=0.2)
# 3
1 - stats.norm.cdf(np.log(90/80), loc=0.16, scale=0.15*np.sqrt(2))
# 4
print(98/95-1)
# 5.a
print((54+0.2)/52-1)
# 5.b
print((59+0.25)/53 * (53+0.2)/54 * (54+0.2)/52 - 1)
# 5.c
print(np.log((53+0.2)/54))
# 7.b
stats.norm.cdf(2, loc=0.24, scale=np.sqrt(1.88))
