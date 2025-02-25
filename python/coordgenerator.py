import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

def generate_random_coordinates_df(spark, num_points, min_lat, max_lat, min_lon, max_lon):
    """
    Generate random geographic coordinates within a specified bounding box and return as a PySpark DataFrame.

    Parameters:
    - spark: An existing SparkSession.
    - num_points: Number of random points to generate.
    - min_lat: Minimum latitude of the bounding box.
    - max_lat: Maximum latitude of the bounding box.
    - min_lon: Minimum longitude of the bounding box.
    - max_lon: Maximum longitude of the bounding box.

    Returns:
    - A PySpark DataFrame with columns 'latitude' and 'longitude'.
    """
    # Generate random coordinates
    random_coordinates = [
        (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))
        for _ in range(num_points)
    ]

    # Define schema
    schema = StructType([
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True)
    ])

    # Create DataFrame
    df = spark.createDataFrame(random_coordinates, schema)

    return df
