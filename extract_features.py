import pandas as pd
import sys
sys.path.append('.')
import quant.features.pipeline as p

df = pd.DataFrame({
    'open': [1]*100, 
    'high':[2]*100, 
    'low':[1]*100, 
    'close':[1.5]*100, 
    'volume':[10]*100, 
    'taker_buy_volume':[5]*100, 
    'taker_sell_volume':[5]*100, 
    'funding_rate_raw':[0]*100, 
    'open_interest':[100]*100, 
    'open_interest_value':[1000]*100, 
    'liquidations_long_vol':[100]*100, 
    'liquidations_short_vol':[50]*100
})
df.index = pd.date_range('2023-01-01', periods=100, freq='H')

import warnings
warnings.filterwarnings('ignore')

df_o = df.copy()
df_o = p.momentum.compute(df_o)
df_o = p.volatility.compute(df_o)
df_o = p.candle_geometry.compute(df_o)
df_o = p.trend.compute(df_o)
df_o = p.volume.compute(df_o)
df_o = p.time_encoding.compute(df_o)
df_o = p.microstructure.compute(df_o)
df_o = p.cross_timeframe.compute(df_o)
df_o = p.order_flow.compute(df_o)
df_o = p.funding_rate.compute(df_o)
df_o = p.open_interest.compute(df_o)
df_o = p.liquidation.compute(df_o)
df_o = p.crypto_session.compute(df_o)

feats = p.get_feature_columns(df_o)
print('===FEATURES===')
print('\n'.join(feats))
print('===ENDFEATURES===')
