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

# File libraries
from importlib.resources import files
import json

# Custom libraries
from .helpers import *

# -----------------------------------------------------------------------------
# Close Encounter Calculator Class
# -----------------------------------------------------------------------------

class CloseEncounters:
    """A class to calculate close encounters between flights."""

    earth_radius_km = 6378  # Approximate Earth radius in kilometers

    def __init__(self, spark):
        # Spark
        self.spark = spark

        # H3 edgelength data
        self.edge_lengths = self._load_h3_edgelengths()

        # Input data
        self.traj_sdf = None
        
        # Resampling data
        self.resampled_sdf = None
        self.resample_freq_s = None
        self.resample_t_max = None
        
        ## Close encounter output data
        self.close_encounter_sdf = None # Spark DataFrame
        self.close_encounter_pdf = None # Pandas DataFrame

        ## Kepler config data
        self.__kepler_config = self._load_kepler_config()

    def load_spark_trajectories(
        self,
        traj_sdf,
        flight_id_col,
        icao24_col,
        longitude_col,
        latitude_col,
        time_over_col,
        flight_level_col
    ):
        for old_col, new_col in {
            'flight_id': flight_id_col,
            'icao24': icao24_col,
            'longitude': longitude_col,
            'latitude': latitude_col,
            'time_over': time_over_col,
            'flight_level': flight_level_col
        }.items():
            traj_sdf = traj_sdf.withColumnRenamed(old_col, new_col)
            
        self.traj_sdf = traj_sdf.select(
            col('flight_id'),
            col('icao24'),
            col('longitude'),
            col('latitude'),
            col('time_over'),
            col('flight_level')
        )
    
        return self

    def load_pandas_trajectories(
        self,
        traj_pdf,
        flight_id_col,
        icao24_col,
        longitude_col,
        latitude_col,
        time_over_col,
        flight_level_col
    ):
        traj_pdf = traj_pdf[[flight_id_col, icao24_col, longitude_col, latitude_col, time_over_col, flight_level_col]]
        traj_sdf = self.spark.createDataFrame(traj_pdf)
        
        return self.load_spark_trajectories(
            traj_sdf = traj_sdf,
            flight_id_col = flight_id_col,
            icao24_col = icao24_col,
            longitude_col = longitude_col,
            latitude_col = latitude_col,
            time_over_col = time_over_col,
            flight_level_col = flight_level_col)

    def _load_sample_trajectories_pdf(self) -> pd.DataFrame:
        """
        Load the sample trajectories dataset bundled with the package.
    
        Returns:
            pd.DataFrame: A DataFrame containing sample flight trajectory data.
        """
        data_path = files("close_encounters.data").joinpath("sample_trajectories.parquet")
        with data_path.open("rb") as f:
            return pd.read_parquet(f)

    def load_sample_trajectories(self):
        """
        Load the sample trajectories dataset bundled with the package into the class instance.
    
        Returns:
            self
        """
        self.load_pandas_trajectories(
            traj_pdf = self._load_sample_trajectories_pdf(),
            flight_id_col = 'FLIGHT_ID',
            icao24_col = 'ICAO24',
            longitude_col = 'LONGITUDE',
            latitude_col = 'LATITUDE',
            time_over_col = 'TIME_OVER',
            flight_level_col = 'FLIGHT_LEVEL'
            )
        return self
    

    def load_sample_trajectories_pdf(self) -> pd.DataFrame:
        """
        Load the sample trajectories dataset bundled with the package.
    
        Returns:
            pd.DataFrame: A DataFrame containing sample flight trajectory data.
        """
        data_path = files("close_encounters.data").joinpath("sample_trajectories.parquet")
        with data_path.open("rb") as f:
            return pd.read_parquet(f)
    
    def _load_h3_edgelengths(self) -> pd.DataFrame:
        """
        Load the H3 edgelengths dataset bundled with the package.
    
        Returns:
            pd.DataFrame: A DataFrame containing H3 edgelengths data.
        """
        data_path = files('close_encounters.data').joinpath('h3_edgelengths.csv')
        with data_path.open("rb") as f:
            return pd.read_csv(f)
    
    def _load_kepler_config(self) -> dict:
        """
        Load the the Kepler.gl config template bundled with the package.
    
        Returns:
            pd.DataFrame: A dictionary containing the Kepler.gl config template.
        """
        data_path = files('close_encounters.resources').joinpath('kepler-config.json')
        with data_path.open("rb") as json_file:
            return json.load(json_file)
    
    def resample(self, freq_s=5, t_max=10, p_numb=100):
        """
        Resample the trajectory data to a fixed frequency and interpolate.

        Args:
            freq_s (int): Resampling frequency in seconds.
            t_max (int): Maximum time interval when data is missing that should be interpolated.
            p_numb (int): Number of partitions to use after resampling.
        """

        if (self.resample_freq_s == freq_s) & (self.resample_t_max == t_max):
            print(f'Skipping resample: Already done (w. freq_s = {freq_s} and t_max = {t_max})')
            return self
        
        freq = f"{freq_s} sec"

        resampled_sdf = TSDF(
            self.traj_sdf,
            ts_col="time_over",
            partition_cols=["flight_id", "icao24"]
        )

        resampled_sdf = resampled_sdf.resample(freq=freq, func="mean").interpolate(
            method='linear',
            freq=freq,
            show_interpolated=True,
            perform_checks=False
        )

        resampled_sdf = resampled_sdf.df.repartition(p_numb, ["flight_id"])
    
        # -----------------------------------------------------------------------------
        # Delete resampled periods which are longer than DeltaT = 10 min
        # -----------------------------------------------------------------------------
    
        # Define a window partitioned by flight and segment and ordered by time
        w = Window.partitionBy("flight_id").orderBy("time_over")
    
        # Flag changes in interpolation status (start of new group)
        resampled_sdf = resampled_sdf.withColumn(
            "interpolation_group_change",
            (F.col("is_ts_interpolated") != F.lag("is_ts_interpolated", 1).over(w)).cast("int")
        )
    
        # Fill nulls in the first row with 1 (new group)
        resampled_sdf = resampled_sdf.withColumn(
            "interpolation_group_change",
            F.when(F.col("interpolation_group_change").isNull(), 1).otherwise(F.col("interpolation_group_change"))
        )
    
        # Create a cumulative sum over the changes to assign group IDs
        resampled_sdf = resampled_sdf.withColumn(
            "interpolation_group_id",
            F.sum("interpolation_group_change").over(w)
        )
    
        # Add min and max timestamp per interpolation group
        group_window = Window.partitionBy("flight_id", "interpolation_group_id")
    
        resampled_sdf = resampled_sdf.withColumn("group_start_time", F.min("time_over").over(group_window))
        resampled_sdf = resampled_sdf.withColumn("group_end_time", F.max("time_over").over(group_window))
    
        # Calculate duration in seconds for each interpolation group
        resampled_sdf = resampled_sdf.withColumn(
            "interpolation_group_duration_sec",
            F.col("group_end_time").cast("long") - F.col("group_start_time").cast("long")
        )
    
        # Filter logic:
        # - If not interpolated, keep
        # - If interpolated, keep only if group duration <= t_max * 60 seconds
        resampled_sdf = resampled_sdf.filter(
            (~F.col("is_ts_interpolated")) |
            ((F.col("is_ts_interpolated")) & (F.col("interpolation_group_duration_sec") <= t_max*60))
        )
    
        # Drop helper columns
        resampled_sdf = resampled_sdf.drop(
            "interpolation_group_change", 
            "interpolation_group_id",
            "group_start_time", 
            "group_end_time", 
            "interpolation_group_duration_sec",
            "is_interpolated_flight_level", 
            "is_interpolated_latitude", 
            "is_interpolated_longitude")
    
        # Add a segment ID
        resampled_sdf = resampled_sdf.withColumn("segment_id", monotonically_increasing_id())
        resampled_sdf = resampled_sdf.repartition(p_numb, ["flight_id", "segment_id"])
        resampled_sdf.cache() # Keep, this is needed to persist the IDs and speed up further calculations
        resampled_sdf.count()

        # Housekeeping
        self.resampled_sdf = resampled_sdf
        self.resample_freq_s = freq_s 
        self.resample_t_max = t_max
        return self

    def find_close_encounters(
        self,
        h_dist_NM = 5, 
        v_dist_FL = 10, 
        v_cutoff_FL = 245, 
        freq_s=5, 
        t_max=10, 
        p_numb=100,
        method='half_disk'
    ):
        if self.traj_sdf is None:
            raise Exception("No trajectories loaded. Use .load_pandas_trajectories or .load_spark_trajectories before trying to find encounters.")
        
        if (method == 'half_disk'):
            self.resample(freq_s=freq_s, t_max=t_max, p_numb=p_numb)

                     # Add H3 index and neighbors
            resampled_w_h3 = self.resampled_sdf.withColumn("h3_index", lat_lon_to_h3_udf(col("latitude"), col("longitude"), lit(resolution)))
            resampled_w_h3 = resampled_w_h3.withColumn("h3_neighbours", get_half_disk_udf(col("h3_index")))
        
            
        
            # -----------------------------------------------------------------------------
            # Explode neighbors and group by time_over and h3_neighbour to collect IDs when there's multiple FLIGHT_ID in a cell
            # -----------------------------------------------------------------------------
            exploded_df = resampled_w_h3.withColumn("h3_neighbour", explode(col("h3_neighbours")))
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
            segment_meta_df = self.resampled.select("segment_id", "flight_level", "flight_id")
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
            coords_sdf1 = self.resampled.withColumnRenamed("segment_id", "ID1") \
                .withColumnRenamed("latitude", "lat1") \
                .withColumnRenamed("longitude", "lon1") \
                .withColumnRenamed("time_over", "time1") \
                .withColumnRenamed("flight_level", 'flight_lvl1') \
                .withColumnRenamed("flight_id", "flight_id1") \
                .withColumnRenamed("icao24", "icao241") \
                .select("ID1", "lat1", "lon1", "time1", "flight_lvl1", "flight_id1", "icao241")
        
            coords_sdf2 = self.resampled.withColumnRenamed("segment_id", "ID2") \
                .withColumnRenamed("latitude", "lat2") \
                .withColumnRenamed("longitude", "lon2") \
                .withColumnRenamed("time_over", "time2") \
                .withColumnRenamed("flight_level", 'flight_lvl2') \
                .withColumnRenamed("flight_id", "flight_id2") \
                .withColumnRenamed("icao24", "icao242") \
                .select("ID2", "lat2", "lon2", "time2", "flight_lvl2", "flight_id2", "icao242")
        
            coords_sdf1 = coords_sdf1.repartition(p_numb, "ID1")
            coords_sdf2 = coords_sdf2.repartition(p_numb, "ID2")
        
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

            df_pairs = df_pairs.withColumn(
                'flight_id1',
                F.concat(F.lit('ID_'), F.round(F.col('flight_id1')).cast('int').cast('string'))
            )
            
            df_pairs = df_pairs.withColumn(
                'flight_id2',
                F.concat(F.lit('ID_'), F.round(F.col('flight_id2')).cast('int').cast('string'))
            )

            df_pairs.cache()
            
            self.close_encounter_sdf = df_pairs
            return self.close_encounter_sdf