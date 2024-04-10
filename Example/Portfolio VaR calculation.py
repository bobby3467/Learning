# Databricks notebook source
query = ''' 
with portfolio as 
(
select instrument_id
        ,left(key_account_id,10) as account_id
from gold.prod_hsc.portfolio_intraday_position_to_market
where 1=1
and key_account_id in ('011PS00001_G3SB','011PQ00001_G3SB')
group by instrument_id,left(key_account_id,10)
)

,t1 as 
(
select  date_code
    ,instrument_id
    ,adjusted_closing_price as price
from gold.prod_hsc.fact_market
where 1=1
and date_code >= date_add(curdate(), -365 ) 
and date_code < curdate()
and instrument_id in (select instrument_id from portfolio)


)
select a.*,b.account_id
from t1 a
left join portfolio b on a.instrument_id = b.instrument_id
'''

# COMMAND ----------

import pandas as pd


df = spark.sql(query)
df = df.toPandas()
equity = df['account_id'].isin(['011PS00001'])
quant = df['account_id'].isin(['011PQ00001'])
df_equity = df[equity]
df_quant = df[quant]

# COMMAND ----------

display(df_quant)

# COMMAND ----------

df_equity = pd.pivot_table(data = df_equity,columns=['instrument_id'],values=['price'],aggfunc='max',index='date_code')
df_equity = df_equity.droplevel(0,axis=1)
df_equity.columns.name = None
df_equity.sort_index()

# COMMAND ----------

df_equity.sort_index(inplace=True)

# COMMAND ----------

df_quant = pd.pivot_table(data = df_quant,columns=['instrument_id'],values=['price'],aggfunc='max',index='date_code')
df_quant = df_quant.droplevel(0,axis=1)
df_quant.columns.name = None
df_quant.sort_index(inplace=True)

# COMMAND ----------

import numpy as np

log_returns_equity = np.log(df_equity/df_equity.shift(1))
log_returns_equity = log_returns_equity.dropna()
display(log_returns_equity)

# COMMAND ----------

log_returns_quant = np.log(df_quant/df_quant.shift(1))
log_returns_quant = log_returns_quant.dropna()
display(log_returns_quant)

# COMMAND ----------

log_returns_quant.cov()

# COMMAND ----------

log_returns_equity_cov = log_returns_equity.cov()
log_returns_quant_cov = log_returns_quant.cov()

# COMMAND ----------

def standard_deviation(weights,cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

# COMMAND ----------

query_portfolio_val = '''
select left(key_account_id,10) as account_id
    ,instrument_id
    ,ending_position
from gold.prod_hsc.portfolio_intraday_position_to_market
where 1=1
and key_account_id in ('011PS00001_G3SB','011PQ00001_G3SB')
'''
portfolio_value = spark.sql(query_portfolio_val)
portfolio_value = portfolio_value.toPandas()
equity_val = portfolio_value['account_id'].isin(['011PS00001'])
quant_val = portfolio_value['account_id'].isin(['011PQ00001'])

# COMMAND ----------

port_value_equity = portfolio_value[equity_val]
port_value_quant = portfolio_value[quant_val]

# COMMAND ----------

equity_port_val = port_value_equity['ending_position'].sum()
quant_port_val = port_value_quant['ending_position'].sum()

# COMMAND ----------

log_returns_quant.std()

# COMMAND ----------

port_value_equity['weights'] = port_value_equity['ending_position'].apply(lambda x: x/equity_port_val)
port_value_equity.set_index('instrument_id',inplace=True)

port_value_quant['weights'] = port_value_quant['ending_position'].apply(lambda x: x/quant_port_val)
port_value_quant.set_index('instrument_id',inplace=True)

# COMMAND ----------

equity_weights = np.array(port_value_equity['weights'])
quant_weights = np.array(port_value_quant['weights'])

# COMMAND ----------

equity_std_dev = standard_deviation(equity_weights,log_returns_equity_cov)
equity_std_dev * np.sqrt(252)

# COMMAND ----------

quant_std_dev = standard_deviation(quant_weights,log_returns_quant_cov)
quant_std_dev * np.sqrt(252)

# COMMAND ----------

equity_weights

# COMMAND ----------

log_returns_equity_cov

# COMMAND ----------

equity_weights.T @ log_returns_equity_cov
