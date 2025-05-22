import json
from keplergl import KeplerGl
from .datasets import *
import pandas as pd

def create_keplergl_html(encounters_df, filename = None):
    keplergl_config = load_kepler_config() 
    # Dynamically modify params 
    data_id = 'Close Encounter Data'
    keplergl_config['config']['visState']['filters'][0]['value'] = [encounters_df.time_over.min().timestamp()*1000, encounters_df.time_over.max().timestamp()*1000]
    keplergl_config['config']['visState']['interactionConfig']['tooltip']['fieldsToShow'][data_id][1]['filterProps']['domain'] = encounters_df['flight_id1'].to_list() 
    keplergl_config['config']['visState']['interactionConfig']['tooltip']['fieldsToShow'][data_id][2]['filterProps']['domain'] = encounters_df['flight_id2'].to_list()

    # Create and output map
    if pd.isnull(filename): 
        filename = f'close_encounters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    kepler_map = KeplerGl(data={data_id: encounters_df}, config=keplergl_config)
    kepler_map.save_to_html(file_name=filename)
    print(f'Outputted Kepler.gl map: {filename}')
    print(f'Note: To view it: download and open with a browser.')
    return kepler_map
    