# 一.Article Summary

The title of the paper is "does the singular value decomposition entropy have predictive power for stock market"



The main idea of this paper is to do principal component analysis on the time series return matrix of index constituent stocks, and then calculate the information entropy of the variance proportion. The increment of this information entropy is related to the price trend of the index.
# 二.Reproduce method

## 1.Data

Select the closing price of CSI 500etf from 2014 to 2021, and the data comes from Mikuang.There is also an all A-share data set and an index component data set, but they were not uploaded due to the limitation of upload size.

## 2.Result

I calculated Sharpe ratio, daily PNL, turnover rate and other indicators. 1bp sliding point, the securities lending rate is 4bps, and the risk-free interest rate is set to 0.