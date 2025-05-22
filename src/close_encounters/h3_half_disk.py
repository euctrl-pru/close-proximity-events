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
from pyspark.sql import Window
from tempo import *

# Data libraries
import h3
import pandas as pd
import numpy as np
from datetime import datetime
from .helpers import *

# -----------------------------------------------------------------------------
# Close encounter default parameters
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Default / Automatic parameters
# -----------------------------------------------------------------------------
def CloseEncountersH3HalfDisk(coords_df, distance_nm = 5, FL_diff = 10, FL_min = 245, deltaT_min = 10, pnumb = 100, spark = None):
    resolution = select_resolution_half_disk(distance_nm)
    earth_radius_km = 6378
    print(f"The selected resolution for a distance of {distance_nm} NM is: {resolution}")

    # -----------------------------------------------------------------------------
    # Load and Filter Data
    # -----------------------------------------------------------------------------
    coords_df = coords_df[coords_df.FLIGHT_LEVEL > FL_min]
    coords_df = coords_df[['FLIGHT_ID', 'LONGITUDE', 'LATITUDE', 'TIME_OVER', 'FLIGHT_LEVEL', 'ICAO24']].rename(
        columns={
            'LATITUDE': 'latitude', 
            'LONGITUDE': 'longitude'
            }
        )
    coords_df.columns = [x.lower() for x in coords_df.columns]

    print(f"Number of rows as input: {coords_df.shape}")
    coords_df = spark.createDataFrame(coords_df)

    # -----------------------------------------------------------------------------
    # Resample and interpolate
    # -----------------------------------------------------------------------------

    coords_df = TSDF(coords_df, ts_col="time_over", partition_cols = ["flight_id", "icao24"])
    coords_df = coords_df.resample(freq="5 sec", func="mean").interpolate(method='linear', freq="5 sec", show_interpolated = True).df
    coords_df = coords_df.repartition(pnumb, ["flight_id"])
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
    coords_df = coords_df.drop(
        "interpolation_group_change", 
        "interpolation_group_id",
        "group_start_time", 
        "group_end_time", 
        "interpolation_group_duration_sec",
        "is_interpolated_flight_level", 
        "is_interpolated_latitude", 
        "is_interpolated_longitude")

    # Add a segment ID
    coords_df = coords_df.withColumn("segment_id", monotonically_increasing_id())
    coords_df = coords_df.repartition(pnumb, ["flight_id", "segment_id"])
    coords_df.cache() # Keep, this is needed to persist the IDs and speed up further calculations
    coords_df.count()

    # Add H3 index and neighbors
    coords_df = coords_df.withColumn("h3_index", lat_lon_to_h3_udf(col("latitude"), col("longitude"), lit(resolution)))
    coords_df = coords_df.withColumn("h3_neighbours", get_half_disk_udf(col("h3_index")))

    

    # -----------------------------------------------------------------------------
    # Explode neighbors and group by time_over and h3_neighbour to collect IDs when there's multiple FLIGHT_ID in a cell
    # -----------------------------------------------------------------------------
    exploded_df = coords_df.withColumn("h3_neighbour", explode(col("h3_neighbours")))
    #print(f"Exploded df nrow = {exploded_df.count()}")

    grouped_df = (exploded_df.groupBy(["time_over", "h3_neighbour"])
                .agg(F.countDistinct("flight_id").alias("flight_count"),
                    F.collect_list("segment_id").alias("id_list"))
                .filter(F.col("flight_count") > 1)
                .drop("flight_count"))

    grouped_df = grouped_df.filter(F.size("id_list") > 1)
    #print(f"Grouped df nrow = {grouped_df.count()}")
    # -----------------------------------------------------------------------------
    # Create pairwise combinations using self-join on indexed exploded DataFrame
    # -----------------------------------------------------------------------------
    # Explode id_list to individual rows 
    df_exploded = grouped_df.withColumn("segment_id", explode("id_list")).drop("id_list")
    
    # Add back the flight_level as it will speed up self-joins
    segment_meta_df = coords_df.select("segment_id", "flight_level", "flight_id")
    df_exploded = df_exploded.join(segment_meta_df, on="segment_id", how="left")

    # Add index within each h3 group
    window_spec = Window.partitionBy(["time_over","h3_neighbour"]).orderBy("segment_id")
    df_indexed = df_exploded.withColumn("idx", F.row_number().over(window_spec))

    # Self-join to form unique unordered ID pairs
    df_pairs = (
        df_indexed.alias("df1")
        .join(
            df_indexed.alias("df2"),
            (F.col("df1.time_over") == F.col("df2.time_over")) &
            (F.abs(F.col("df1.flight_level") - F.col("df2.flight_level")) < FL_diff) &
            (F.col("df1.h3_neighbour") == F.col("df2.h3_neighbour")) &
            (F.col("df1.idx") < F.col("df2.idx"))
        )
        .select(
            F.col("df1.time_over").alias("time_over"),
            F.col("df1.h3_neighbour").alias("h3_group"),
            F.col("df1.segment_id").alias("ID1"),
            F.col("df2.segment_id").alias("ID2")
        )
    )

    # -----------------------------------------------------------------------------
    # Clean Pairs, Create Unique Pair ID
    # -----------------------------------------------------------------------------
    df_pairs = df_pairs.filter(col("ID1") != col("ID2")) # should not be necessary as we join on < not <=
    df_pairs = df_pairs.withColumn(
        "ID",
        F.concat_ws("_", F.array_sort(F.array(col("ID1"), col("ID2"))))
    )

    # Define a window partitioned by ID, ordering arbitrarily (or by some column if needed)
    window_spec = Window.partitionBy("ID").orderBy(F.monotonically_increasing_id())

    # Add row number to each partition
    df_pairs = df_pairs.withColumn("row_num", F.row_number().over(window_spec))

    # Keep only the first row per ID
    df_pairs = df_pairs.filter(F.col("row_num") == 1).drop("row_num")

    # -----------------------------------------------------------------------------
    # Join with Original Coordinates for Each ID
    # -----------------------------------------------------------------------------
    coords_sdf1 = coords_df.withColumnRenamed("segment_id", "ID1") \
        .withColumnRenamed("latitude", "lat1") \
        .withColumnRenamed("longitude", "lon1") \
        .withColumnRenamed("time_over", "time1") \
        .withColumnRenamed("flight_level", 'flight_lvl1') \
        .withColumnRenamed("flight_id", "flight_id1") \
        .withColumnRenamed("icao24", "icao241") \
        .select("ID1", "lat1", "lon1", "time1", "flight_lvl1", "flight_id1", "icao241")

    coords_sdf2 = coords_df.withColumnRenamed("segment_id", "ID2") \
        .withColumnRenamed("latitude", "lat2") \
        .withColumnRenamed("longitude", "lon2") \
        .withColumnRenamed("time_over", "time2") \
        .withColumnRenamed("flight_level", 'flight_lvl2') \
        .withColumnRenamed("flight_id", "flight_id2") \
        .withColumnRenamed("icao24", "icao242") \
        .select("ID2", "lat2", "lon2", "time2", "flight_lvl2", "flight_id2", "icao242")

    coords_sdf1 = coords_sdf1.repartition(pnumb, "ID1")
    coords_sdf2 = coords_sdf2.repartition(pnumb, "ID2")

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
    #df_pairs.cache()

    df_pairs = df_pairs.withColumn("lat1_rad", radians(col("lat1"))) \
                   .withColumn("lat2_rad", radians(col("lat2"))) \
                   .withColumn("lon1_rad", radians(col("lon1"))) \
                   .withColumn("lon2_rad", radians(col("lon2")))

    df_pairs = df_pairs.withColumn(
        "distance_nm",
        0.539957 * 2 * earth_radius_km * atan2(
            sqrt(
                (sin(col('lat2_rad') - col('lat1_rad')) / 2)**2 +
                cos(col('lat1_rad')) * cos(col('lat2_rad')) *
                (sin(col('lon2_rad') - col('lon1_rad')) / 2)**2
            ),
            sqrt(1 - (
                (sin(col('lat2_rad') - col('lat1_rad')) / 2)**2 +
                cos(col('lat1_rad')) * cos(col('lat2_rad')) *
                (sin(col('lon2_rad') - col('lon1_rad')) / 2)**2
            ))
        )
    )

    df_pairs = df_pairs.drop('lat1_rad', 'lat2_rad', 'lon1_rad', 'lon2_rad')

    df_pairs = df_pairs.filter(col('distance_nm') <= lit(distance_nm))

    # -----------------------------------------------------------------------------
    # Fetch sample
    # -----------------------------------------------------------------------------

    #df_pairs.cache()
    df = df_pairs.toPandas()
    df['flight_id1'] = df['flight_id1'].apply(lambda l: f'ID_{round(l)}')
    df['flight_id2'] = df['flight_id2'].apply(lambda l: f'ID_{round(l)}')
    print(f"Number of unique ID pairs: {df.shape[0]}")
    return df


