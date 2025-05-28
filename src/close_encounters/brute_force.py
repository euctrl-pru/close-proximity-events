# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# PySpark libraries
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import (
    udf, col, explode, radians,
    sin, cos, sqrt, atan2, lit, monotonically_increasing_id
)
from pyspark.sql.types import (
    StringType, ArrayType )
from pyspark.sql import Window
from tempo import *

# Data libraries
import h3
import pandas as pd
import numpy as np
from datetime import datetime

# Custom libraries
from .helpers import select_resolution

# -----------------------------------------------------------------------------
# Close encounter default parameters
# -----------------------------------------------------------------------------
# Minimal horizontal distance before close encounter (NM)
distance_nm = 5

# Minimal vertical distance before close encounter (flight levels - FL)
FL_diff = 9

# The minimum flight level for assessment of the trajectory (lower sections are not analyzed)
FL_min = 250

# The maximum period we should interpolate in case of missing state-vectors (deltaT in minutes)
deltaT_min = 10

# -----------------------------------------------------------------------------
# Default / Automatic parameters
# -----------------------------------------------------------------------------
def close_encounters_brute_force(coords_df, distance_nm = 5, FL_diff = 9, FL_min = 250, deltaT_min = 10, spark = None):
    resolution = select_resolution(distance_nm)
    earth_radius_km = 6378
    print(f"The selected resolution for a distance of {distance_nm} NM is: {resolution}")

    # -----------------------------------------------------------------------------
    # Load and Filter Data
    # -----------------------------------------------------------------------------
    coords_df = coords_df[coords_df.FLIGHT_LEVEL > FL_min]
    coords_df = coords_df[['FLIGHT_ID', 'LONGITUDE', 'LATITUDE', 'TIME_OVER', 'FLIGHT_LEVEL']].rename(
        columns={
            'LATITUDE': 'latitude', 
            'LONGITUDE': 'longitude'
            }
        )
    coords_df.columns = [x.lower() for x in coords_df.columns]

    print(f"Number of rows as input: {coords_df.shape}")
    coords_df = spark.createDataFrame(coords_df)
    #coords_df = coords_df.limit(100)
    # -----------------------------------------------------------------------------
    # Resample and interpolate
    # -----------------------------------------------------------------------------

    coords_df = TSDF(coords_df, ts_col="time_over", partition_cols = ["flight_id"])
    coords_df = coords_df.resample(freq="5 sec", func="mean").interpolate(method='linear', freq="5 sec", show_interpolated = True).df
    coords_df = coords_df.repartition(100, ["flight_id"])
    #print(f"Number of rows after resamplin and interpolating: {coords_df.count()}")

    # -----------------------------------------------------------------------------
    # Delete resampled periods which are longer than DeltaT = 10 min
    # -----------------------------------------------------------------------------

    # Define a window partitioned by flight and segment and ordered by time
    w = Window.partitionBy("flight_id").orderBy("time_over")

    # Flag changes in interpolation status (start of new group)
    coords_df = coords_df.withColumn(
        "interpolation_group_change",
        (F.col("is_ts_interpolated") != F.lag("is_ts_interpolated", 1).over(w)).cast("int")
    )

    # Fill nulls in the first row with 1 (new group)
    coords_df = coords_df.withColumn(
        "interpolation_group_change",
        F.when(F.col("interpolation_group_change").isNull(), 1).otherwise(F.col("interpolation_group_change"))
    )

    # Create a cumulative sum over the changes to assign group IDs
    coords_df = coords_df.withColumn(
        "interpolation_group_id",
        F.sum("interpolation_group_change").over(w)
    )

    # Add min and max timestamp per interpolation group
    group_window = Window.partitionBy("flight_id", "interpolation_group_id")

    coords_df = coords_df.withColumn("group_start_time", F.min("time_over").over(group_window))
    coords_df = coords_df.withColumn("group_end_time", F.max("time_over").over(group_window))

    # Calculate duration in seconds for each interpolation group
    coords_df = coords_df.withColumn(
        "interpolation_group_duration_sec",
        F.col("group_end_time").cast("long") - F.col("group_start_time").cast("long")
    )

    # Filter logic:
    # - If not interpolated, keep
    # - If interpolated, keep only if group duration <= deltaT_min * 60 seconds
    coords_df = coords_df.filter(
        (~F.col("is_ts_interpolated")) |
        ((F.col("is_ts_interpolated")) & (F.col("interpolation_group_duration_sec") <= deltaT_min*60))
    )

    # Drop helper columns
    coords_df = coords_df.drop("interpolation_group_change", "interpolation_group_id",
                            "group_start_time", "group_end_time", "interpolation_group_duration_sec")

    # Add a segment ID
    coords_df = coords_df.withColumn("segment_id", monotonically_increasing_id())
    coords_df = coords_df.repartition(100, ["flight_id", "segment_id"])

    #coords_df = coords_df.filter(col('time_over')==datetime(2024,7,1,12,1,0))
    coords_df.cache() # Keep, this is needed to persist the IDs
    coords_df.count()

    # -----------------------------------------------------------------------------
    # Create pairwise combinations using self-join on indexed exploded DataFrame
    # -----------------------------------------------------------------------------
    # Explode id_list to individual rows and add index within each h3 group
    
    coords_df_f = coords_df.select(['segment_id','time_over'])
    window_spec = Window.partitionBy("time_over").orderBy("segment_id")
    df_indexed = coords_df_f.withColumn("idx", F.row_number().over(window_spec))

    # Self-join to form unique unordered ID pairs
    df_pairs = (
        df_indexed.alias("df1")
        .join(
            df_indexed.alias("df2"),
            (F.col("df1.time_over") == F.col("df2.time_over")) &
            (F.col("df1.idx") < F.col("df2.idx"))
        )
        .select(
            F.col("df1.segment_id").alias("ID1"),
            F.col("df2.segment_id").alias("ID2")
        )
    )
    #df_pairs.cache()
    #print(f"Number of generated pairs: {df_pairs.count()}")
    # -----------------------------------------------------------------------------
    # Clean Pairs, Create Unique Pair ID
    # -----------------------------------------------------------------------------
    df_pairs = df_pairs.filter(col("ID1") != col("ID2")) # should not be necessary as we join on < not <=
    df_pairs = df_pairs.withColumn(
        "ID",
        F.concat_ws("_", F.array_sort(F.array(col("ID1"), col("ID2"))))
    )

    # -----------------------------------------------------------------------------
    # Join with Original Coordinates for Each ID
    # -----------------------------------------------------------------------------
    coords_sdf1 = coords_df.withColumnRenamed("segment_id", "ID1") \
        .withColumnRenamed("latitude", "lat1") \
        .withColumnRenamed("longitude", "lon1") \
        .withColumnRenamed("time_over", "time1") \
        .withColumnRenamed("flight_level", 'flight_lvl1') \
        .withColumnRenamed("flight_id", "flight_id1") \
        .select("ID1", "lat1", "lon1", "time1", "flight_lvl1", "flight_id1")

    coords_sdf2 = coords_df.withColumnRenamed("segment_id", "ID2") \
        .withColumnRenamed("latitude", "lat2") \
        .withColumnRenamed("longitude", "lon2") \
        .withColumnRenamed("time_over", "time2") \
        .withColumnRenamed("flight_level", 'flight_lvl2') \
        .withColumnRenamed("flight_id", "flight_id2") \
        .select("ID2", "lat2", "lon2", "time2", "flight_lvl2", "flight_id2")

    coords_sdf1 = coords_sdf1.repartition(100, "ID1")
    coords_sdf2 = coords_sdf2.repartition(100, "ID2")

    df_pairs = df_pairs.join(coords_sdf1, on="ID1", how="left")
    df_pairs = df_pairs.join(coords_sdf2, on="ID2", how="left")
    #df_pairs.cache()
    #print(f"Number of pairs (raw): {df_pairs.count()}")
    # -----------------------------------------------------------------------------
    # Calculate and filter based on time differense (s)
    # -----------------------------------------------------------------------------
    df_pairs = df_pairs.withColumn('time_diff_s', F.unix_timestamp(F.col("time1")) - F.unix_timestamp(F.col("time2")))
    df_pairs = df_pairs.filter(F.abs(F.col('time_diff_s')) == 0)
    #df_pairs.cache()
    #print(f"Number of pairs after time filter {df_pairs.count()}")
    # -----------------------------------------------------------------------------
    # Calculate and filter based on height differense (s)
    # -----------------------------------------------------------------------------
    df_pairs = df_pairs.withColumn('FL_diff', F.col("flight_lvl1") - F.col("flight_lvl2"))
    df_pairs = df_pairs.filter(F.abs(F.col('FL_diff')) < lit(FL_diff))
    #df_pairs.cache()
    #print(f"Number of pairs after FL filter {df_pairs.count()}")

    # -----------------------------------------------------------------------------
    # Calulate and filter based on distance (km)
    # -----------------------------------------------------------------------------
    df_pairs = df_pairs.withColumn(
        "distance_nm",
        0.539957 * 2 * earth_radius_km * atan2(
            sqrt(
                (sin(radians(col("lat2")) - radians(col("lat1"))) / 2)**2 +
                cos(radians(col("lat1"))) * cos(radians(col("lat2"))) *
                (sin(radians(col("lon2")) - radians(col("lon1"))) / 2)**2
            ),
            sqrt(1 - (
                (sin(radians(col("lat2")) - radians(col("lat1"))) / 2)**2 +
                cos(radians(col("lat1"))) * cos(radians(col("lat2"))) *
                (sin(radians(col("lon2")) - radians(col("lon1"))) / 2)**2
            ))
        )
    )

    df_pairs = df_pairs.filter(col('distance_nm') <= lit(distance_nm))

    # # -----------------------------------------------------------------------------
    # # Fetch sample
    # # -----------------------------------------------------------------------------

    df = df_pairs.toPandas()
    print(f"Number of unique ID pairs: {df.shape[0]}")
    return df