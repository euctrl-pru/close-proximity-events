import pandas as pd
import numpy as np

h3_df = pd.read_csv('~/Repos/close-encounters/data/edgelengths.csv') 

h3_df = h3_df[['res', 'Min Edge Length km (Hex)']]
h3_df = h3_df.rename({'Min Edge Length km (Hex)':'min_edge_length_km'}, axis = 1)
h3_df['min_edge_length_NM'] = h3_df['min_edge_length_km'] / 1.852
h3_df['3xd_max'] = h3_df['min_edge_length_NM']*3*np.sqrt(3)/2
h3_df['3xd_min'] = h3_df['3xd_max'].shift(-1).fillna(0)

def select_resolution(delta_x_nm, h3_df = h3_df):
    h3_df_ = h3_df[np.logical_and(h3_df['3xd_max'] > delta_x_nm, h3_df['3xd_min'] <= delta_x_nm)]
    return int(h3_df_.res.values[0])