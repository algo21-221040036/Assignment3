import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#1bp滑点
#融券利率为4bps
#无风险利率设为0
n1 = 22
n2 = 5
nameIndex = 'zz500'
startDate = 20160101
endDate = 20210101
#获取行情数据
mkt = pd.read_csv(r'.\daily_bar_stock.csv')
mkt = mkt.loc[mkt['Date']>=startDate,:]
mkt = mkt.loc[mkt['Date']<=endDate,:]
mkt = mkt.set_index(['Date', 'SecurityIDExtend']).sort_index()

status = mkt.Status.unstack()
mask = ~((status == 0)|(status.shift() == 0))
pricePost = mkt['LastPrice'].unstack() * mkt.exFactor.unstack().cumprod()
rtSym = pricePost.pct_change()
rtSym = rtSym[mask]
# 获取指数成分权重
weightIndex = pd.read_csv(r'.\weight_index_%s.csv'%nameIndex)
weightIndex = weightIndex.rename({'date':'Date', 'codes': 'SecurityIDExtend'}, axis=1)
weightIndex.SecurityIDExtend = weightIndex.SecurityIDExtend.map(lambda x: x.replace('SH', 'XSHG').replace('SZ', 'XSHE'))
weightIndex = weightIndex.set_index(['Date', 'SecurityIDExtend']).sort_index().weight
datesInWeightIndex = weightIndex.index.get_level_values(level='Date').drop_duplicates()

etfPrice = pd.read_csv(r'C:\Users\Administrator\Desktop\实习\zz500etf')
etfPrice['return_past_5'] = etfPrice['close'] / etfPrice['close'].shift(-5) - 1
etfPrice['date'] = etfPrice['date'].str.replace('-','').astype(int)
etfPrice = etfPrice.set_index(['date'])

dwd = pd.Series(name='wd')
for i, d in enumerate(rtSym.index):
    if i>n1+n2:
        latestDateInW = datesInWeightIndex.asof(d)
        w = weightIndex.loc[latestDateInW]
        rtSymTmp = rtSym.reindex(w.index, axis=1).fillna(0)
        r = PCA().fit(rtSymTmp.iloc[i-n1-n2+1:i-n2+1]).explained_variance_ratio_
        wd1 = -1 * (r*np.log(r)).sum()
        r = PCA().fit(rtSymTmp.iloc[i-n1-n2+1:i+1]).explained_variance_ratio_
        wd2 = -1 * (r*np.log(r)).sum()
        dwd.loc[d] = wd2-wd1
        print(d,wd2 - wd1)

dwd = pd.concat([dwd,np.sign(etfPrice[['return_past_5']])],axis=1)
dwd = dwd.dropna()
dwd['wd'] = dwd['return_past_5']*dwd['wd']*(-1)#调整+1还是-1       
dwd['wd'] = dwd['wd'].shift(1)

w = pd.concat([dwd['wd'],etfPrice],axis=1).dropna()
w['etf_return'] = w['close'].pct_change()
vol=1
w['wd_position'] = w['wd'].rolling(5).sum()
w = w.dropna()
w['wd-'] = w['wd_position'].where(w['wd_position']<0,0) #分出做空的
w['wd+'] = w['wd_position'].where(w['wd_position']>0,0) #分出做多的 

w['position+'] = (w['wd+']/vol)*100 #单位是万
#做多部分的价值
w['dposition'] = w['position+'].diff()
w['sell_position'] = w['dposition'].where(w['dposition']<0,0).abs()
w['buy_position'] = w['dposition'].where(w['dposition']>0,0)
del w['dposition']
w['sell_return'] = ((w['vwap']*(1-0.001))/w['close'].shift(1))-1
w['buy_return'] = (w['close']/(w['vwap']*(1+0.001)))-1
w['new_value']=w['sell_position']*w['sell_return']+w['buy_position']*w['buy_return']
del w['sell_position'],w['sell_return'],w['buy_return']
w['hold_return'] = (w['close']/w['close'].shift(1))-1
w['hold_position'] = w['position+']-w['buy_position']
del w['buy_position']
w['old_value'] = w['hold_position']*w['hold_return']
w['value+'] = w['old_value']+w['new_value']#做多部分的价值设为value+
del w['new_value'],w['old_value']
del w['hold_return'],w['hold_position']

#做空部分的价值
w['position-'] = (w['wd-']/vol)*100#单位是万
w['dposition'] = w['position-'].diff()
w['sell_position'] = w['dposition'].where(w['dposition']<0,0).abs()
w['buy_position'] = w['dposition'].where(w['dposition']>0,0)
w['sell_return'] = ((w['vwap']*(1-0.001))/w['close']) -1
w['buy_return'] =( w['close'].shift(1)/(w['vwap']*(1+0.001)))-1
w['new_value'] = w['sell_position']*w['sell_return']+w['buy_position']*w['buy_return']
w['hold_return'] = (2-w['close']/w['close'].shift(1))*(1-0.004)-1
w['hold_position'] = (w['position-']+w['sell_position']).abs()
w['old_value'] = w['hold_position']*w['hold_return']
w['value-'] = w['old_value']+w['new_value']#做空部分的价值设为value-

w = w.drop(columns=['dposition','sell_position','buy_position','sell_return',
'buy_return','new_value','hold_return','hold_position','old_value'])

w['pnl'] = w['value+']+w['value-']
w['return'] = w['pnl']/(w['pnl'].std()*math.sqrt(252)*10)
w['position'] = w['position+']+w['position-']
sharpe_rate = (math.sqrt(252)*w['return'].mean())/(w['return'].std()) #年化夏普比
turnover = 0.5*w['position'].diff().abs().mean()/w['position'].abs().mean()#换手率
result = {'hist_signals':w['wd'],'hist_pnl':w['pnl'], 'sharpe':sharpe_rate, 
        'turn_over':turnover,'return':w['return'],'close':w['close'],
        'position':w['position'],'vol':vol,'wd_position':w['wd_position']}

result['hist_pnl'].reset_index()['pnl'].cumsum().plot(title='pnl_cumsum')
plt.show()
print('夏普比为：', result['sharpe'])
print('换手率为：', result['turn_over'])


