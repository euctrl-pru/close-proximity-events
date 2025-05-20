import pandas as pd
import numpy as np
import h3
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    StringType, ArrayType )

h3_df = pd.read_csv('~/Repos/close-encounters/data/edgelengths.csv') 

h3_df = h3_df[['res', 'Min Edge Length km (Hex)']]
h3_df = h3_df.rename({'Min Edge Length km (Hex)':'min_edge_length_km'}, axis = 1)
h3_df['min_edge_length_NM'] = h3_df['min_edge_length_km'] / 1.852
h3_df['3xd_max'] = h3_df['min_edge_length_NM']*3*np.sqrt(3)/2
h3_df['3xd_min'] = h3_df['3xd_max'].shift(-1).fillna(0)

def select_resolution(delta_x_nm, h3_df = h3_df):
    h3_df_ = h3_df[np.logical_and(h3_df['3xd_max'] > delta_x_nm, h3_df['3xd_min'] <= delta_x_nm)]
    return int(h3_df_.res.values[0])

def select_resolution_half_disk(delta_x_nm, h3_df = h3_df):
    h3_df_ = h3_df[h3_df.min_edge_length_km > delta_x_nm]
    return int(max(h3_df_.res.to_list()))


# -----------------------------------------------------------------------------
# Define UDFs for H3
# -----------------------------------------------------------------------------
def lat_lon_to_h3(lat, lon, resolution):
    return h3.latlng_to_cell(lat, lon, resolution)

def grid_disk_k1(cell):
    return h3.grid_disk(cell, k=1)

lat_lon_to_h3_udf = udf(lat_lon_to_h3, StringType())

def get_half_disk(h3_index):
    """
    Returns H3 indices of the three 'northern' neighbors of a given hex,
    interpreted via local IJ coordinates.

    Parameters:
    h3_index (str): H3 index of the origin hex.

    Returns:
    list of str: H3 indices of neighboring cells in IJ directions (1,0), (0,1), (-1,1).
    """
    try:
        i, j = h3.cell_to_local_ij(h3_index, h3_index)
        return [
            h3_index, 
            h3.local_ij_to_cell(h3_index, i, j-1), 
            h3.local_ij_to_cell(h3_index, i+1, j), 
            h3.local_ij_to_cell(h3_index, i+1, j+1)]

    except Exception:
        print(f'[Warning] Could not create disk neighbours for index: {h3_index}')
        return [None, None, None]

# Define the UDF
get_half_disk_udf = udf(get_half_disk, ArrayType(StringType()))