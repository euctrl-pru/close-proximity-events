import pandas as pd

def select_resolution(distance):
    """
    Select the resolution with the largest edge length that is just larger than the given distance using pandas.

    Parameters:
    - distance: The given distance in kilometers.

    Returns:
    - The selected resolution level.
    """
    # Create a DataFrame from the list of tuples (resolution, average edge length in km)
    data = {
        'resolution': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'edge_length': [
            1281.256011, 483.0568391, 182.5129565, 68.97922179, 26.07175968,
            9.854090990, 3.724532667, 1.406475763, 0.531414010, 0.200786148,
            0.075863783, 0.028663897, 0.010830188, 0.004092010, 0.001546100,
            0.000584169
        ]
    } # refine using https://observablehq.com/@nrabinowitz/h3-area-stats
    df = pd.DataFrame(data)

    # Filter the DataFrame to find the minimum edge length greater than the given distance
    valid_resolutions = df[df['edge_length'] > distance/2]

    # Select the resolution with the minimum edge length from the valid resolutions
    selected_resolution = valid_resolutions.loc[valid_resolutions['edge_length'].idxmin(), 'resolution']

    return selected_resolution