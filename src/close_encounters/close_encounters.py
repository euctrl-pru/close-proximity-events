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
    DoubleType, IntegerType, LongType, TimestampType)
from pyspark.sql import Window, DataFrame
from tempo import *

# Data libraries
import h3
import pandas as pd
import numpy as np
from keplergl import KeplerGl
from datetime import datetime
import duckdb
import tempfile
import os

# File libraries
from importlib.resources import files
import json

# Custom libraries
from .helpers import *

# Logging 
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Or logging.DEBUG for more verbosity

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# Close Encounter Calculator Class
# -----------------------------------------------------------------------------

class CloseEncounters:
    """A class to calculate close encounters between flights."""

    earth_radius_km = 6378  # Approximate Earth radius in kilometers

    def __init__(self, spark):
        """
        Initialize the CloseEncounters class.
        
        Args:
            spark (SparkSession): An active Spark session.
        """
        logger.info("Initialized CloseEncounters class.")
        
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
        """
        Load flight trajectory data from a Spark DataFrame.
        
        Args:
            traj_sdf (DataFrame): Spark DataFrame containing trajectory data.
            flight_id_col (str): Column name for flight ID.
            icao24_col (str): Column name for ICAO24 aircraft ID.
            longitude_col (str): Column name for longitude.
            latitude_col (str): Column name for latitude.
            time_over_col (str): Column name for timestamp.
            flight_level_col (str): Column name for flight level.
        
        Returns:
            CloseEncounters: Self instance for method chaining.
        """
        
        for old_col, new_col in {
            'flight_id': flight_id_col,
            'icao24': icao24_col,
            'longitude': longitude_col,
            'latitude': latitude_col,
            'time_over': time_over_col,
            'flight_level': flight_level_col
        }.items():
            traj_sdf = traj_sdf.withColumnRenamed(old_col, new_col)
        
        # Cast time_over to timestamp type
        traj_sdf = traj_sdf.withColumn("time_over", F.to_utc_timestamp(F.col("time_over"), "UTC"))

        self.traj_sdf = traj_sdf.select(
            F.col('flight_id'),
            F.col('icao24'),
            F.col('longitude'),
            F.col('latitude'),
            F.col('time_over'),
            F.col('flight_level')
        )

        logger.info("Loaded trajectory data from Spark DataFrame.")
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
        """
        Load flight trajectory data from a pandas DataFrame.
        
        Args:
            traj_pdf (pd.DataFrame): Pandas DataFrame containing trajectory data.
            flight_id_col (str): Column name for flight ID.
            icao24_col (str): Column name for ICAO24 aircraft ID.
            longitude_col (str): Column name for longitude.
            latitude_col (str): Column name for latitude.
            time_over_col (str): Column name for timestamp.
            flight_level_col (str): Column name for flight level.
        
        Returns:
            CloseEncounters: Self instance for method chaining.
        """
        traj_pdf = traj_pdf[[flight_id_col, icao24_col, longitude_col, latitude_col, time_over_col, flight_level_col]]
        traj_sdf = self.spark.createDataFrame(traj_pdf)
        
        logger.info("Loaded trajectory data from pandas DataFrame.")
        return self.load_spark_trajectories(
            traj_sdf = traj_sdf,
            flight_id_col = flight_id_col,
            icao24_col = icao24_col,
            longitude_col = longitude_col,
            latitude_col = latitude_col,
            time_over_col = time_over_col,
            flight_level_col = flight_level_col)

    def load_parquet_trajectories(
        self,
        parquet_path: str,
        flight_id_col: str,
        icao24_col: str,
        longitude_col: str,
        latitude_col: str,
        time_over_col: str,
        flight_level_col: str
    ):
        """
        Load trajectory data from a Parquet file(path).
        
        Args:
            parquet_path (str): Path to the Parquet file.
            flight_id_col (str): Column name for flight ID.
            icao24_col (str): Column name for ICAO24 aircraft ID.
            longitude_col (str): Column name for longitude.
            latitude_col (str): Column name for latitude.
            time_over_col (str): Column name for timestamp.
            flight_level_col (str): Column name for flight level.
        
        Returns:
            CloseEncounters: Self instance for method chaining.
        """
        traj_sdf = self.spark.read.parquet(parquet_path)
        
        traj_sdf = (
            traj_sdf
            .withColumn(flight_id_col, F.col(flight_id_col).cast(IntegerType()))
            .withColumn(latitude_col, F.col(latitude_col).cast(DoubleType()))
            .withColumn(longitude_col, F.col(longitude_col).cast(DoubleType()))
            .withColumn(flight_level_col, F.col(flight_level_col).cast(IntegerType()))
            .withColumn(time_over_col, F.col(time_over_col).cast(TimestampType()))
        )

        logger.info(f"Loaded trajectory data from parquet: {parquet_path}")
        return self.load_spark_trajectories(
            traj_sdf=traj_sdf,
            flight_id_col=flight_id_col,
            icao24_col=icao24_col,
            longitude_col=longitude_col,
            latitude_col=latitude_col,
            time_over_col=time_over_col,
            flight_level_col=flight_level_col
        )
    
    def _load_sample_trajectories_pdf(self, nrows = None) -> pd.DataFrame:
        """
        Load the bundled sample trajectories as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame containing sample flight trajectory data.
        """
        data_path = files("close_encounters").joinpath("data/sample_trajectories.parquet")
        with data_path.open("rb") as f:
            logger.info("Loaded bundled sample trajectories.")
            if nrows is None:
                return pd.read_parquet(f)
            else:
                return pd.read_parquet(f).iloc[:nrows]

    def load_sample_trajectories(self, nrows = None):
        """
        Load the bundled sample trajectories into the class instance.
        
        Returns:
            CloseEncounters: Self instance for method chaining.
        """
        self.load_pandas_trajectories(
            traj_pdf = self._load_sample_trajectories_pdf(nrows = nrows),
            flight_id_col = 'FLIGHT_ID',
            icao24_col = 'ICAO24',
            longitude_col = 'LONGITUDE',
            latitude_col = 'LATITUDE',
            time_over_col = 'TIME_OVER',
            flight_level_col = 'FLIGHT_LEVEL'
            )
        return self

    def _select_resolution_half_disk(self, h_dist_NM = 5):
        """
        Select the highest H3 resolution such that the minimum edge length exceeds the given distance in order to detect all pairs correctly.
        
        Args:
            h_dist_NM (float): Horizontal threshold in nautical miles.
        
        Returns:
            int: Selected H3 resolution.
        """
        
        h3_df = self.edge_lengths
        h3_df = h3_df[['res', 'Min Edge Length km (Hex)']]
        h3_df = h3_df.rename({'Min Edge Length km (Hex)':'min_edge_length_km'}, axis = 1)
        h3_df['min_edge_length_NM'] = h3_df['min_edge_length_km'] / 1.852
        h3_df_ = h3_df[h3_df.min_edge_length_NM > h_dist_NM]
        return int(max(h3_df_.res.to_list()))
    
    def _load_h3_edgelengths(self) -> pd.DataFrame:
        """
        Load the bundled H3 edgelength reference table.
        
        Returns:
            pd.DataFrame: DataFrame containing H3 resolution and edge lengths.
        """
        data_path = files('close_encounters').joinpath('data/h3_edgelengths.csv')
        with data_path.open("rb") as f:
            return pd.read_csv(f)
    
    def _load_kepler_config(self) -> dict:
        """
        Load the bundled Kepler.gl configuration template.
        
        Returns:
            dict: Dictionary containing Kepler.gl configuration settings.
        """
        data_path = files('close_encounters').joinpath('resources/kepler-config.json')
        with data_path.open("rb") as json_file:
            return json.load(json_file)
    
    def resample(self, freq_s=5, t_max=10, p_numb=100):
        """
        Resample the trajectory data at a fixed interval and linearly interpolate missing values.
        
        Args:
            freq_s (int): Resampling frequency in seconds.
            t_max (int): Maximum allowed interpolation gap in minutes.
            p_numb (int): Number of Spark partitions for output DataFrame.
        
        Returns:
            CloseEncounters: Self instance for method chaining.
        """
        if (self.resample_freq_s == freq_s) & (self.resample_t_max == t_max):
            logger.info(f"Skipping resample: already done (freq_s={freq_s}, t_max={t_max})")
            return self
        
        freq = f"{freq_s} sec"
        traj_sdf = self.traj_sdf.withColumn('flight_level_ft', F.col('flight_level')*100)
        traj_sdf = traj_sdf.drop(traj_sdf.flight_level)        
        
        resampled_sdf = TSDF(
            traj_sdf,
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
            ((F.col("is_ts_interpolated")) &(F.col("interpolation_group_duration_sec") <= t_max*60))
        )
    
        # Drop helper columns
        resampled_sdf = resampled_sdf.drop(
            "interpolation_group_change", 
            "interpolation_group_id",
            "group_start_time", 
            "group_end_time", 
            "interpolation_group_duration_sec",
            "is_interpolated_flight_level_ft", 
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

        logger.info("Resampling complete. Total segments: %d", self.resampled_sdf.count())
        return self

    def find_close_encounters(
        self,
        h_dist_NM = 5, 
        v_dist_ft = 1000, 
        v_cutoff_FL = 245, 
        freq_s=5, 
        t_max=10, 
        p_numb=100,
        method='half_disk',
        output = 'self'
    ):
        """
        Detect spatial and temporal close encounters between aircraft.
        
        Args:
            h_dist_NM (float): Horizontal distance threshold in nautical miles.
            v_dist_ft (float): Vertical distance threshold in feet.
            v_cutoff_FL (int): Minimum flight level cutoff to include.
            freq_s (int): Resampling frequency in seconds.
            t_max (int): Maximum interpolation gap in minutes.
            p_numb (int): Number of Spark partitions to use.
            method (str): Proximity detection method (currently supports 'half_disk').
            output (str): What should be outputted: 'self' (the object itself) / 'pdf' (pandas ce df) / 'sdf' (spark ce df) 
        
        Returns:
            DataFrame: Spark DataFrame containing detected close encounters.
        """
        logger.info("Starting close encounter detection with method='%s'", method)
        
        # Check if data is there
        if self.traj_sdf is None:
            raise Exception("No trajectories loaded. Use .load_pandas_trajectories or .load_spark_trajectories before trying to find encounters.")

        self.resample(freq_s=freq_s, t_max=t_max, p_numb=p_numb)
        
        if (method == 'half_disk'):
            
            resolution = self._select_resolution_half_disk(h_dist_NM = h_dist_NM)

            # Cutoff 
            resampled_co = self.resampled_sdf.filter(F.col('flight_level_ft') >= lit(v_cutoff_FL)*100)
            
            # Add H3 index and neighbors
            resampled_w_h3 = resampled_co.withColumn("h3_index", lat_lon_to_h3_udf(F.col("latitude"), F.col("longitude"), lit(resolution)))
            resampled_w_h3 = resampled_w_h3.withColumn("h3_neighbours", get_half_disk_udf(F.col("h3_index")))
        
            # -----------------------------------------------------------------------------
            # Explode neighbors and group by time_over and h3_neighbour to collect IDs when there's multiple FLIGHT_ID in a cell
            # -----------------------------------------------------------------------------
            exploded_df = resampled_w_h3.withColumn("h3_neighbour", explode(F.col("h3_neighbours")))
            grouped_df = (exploded_df.groupBy(["time_over", "h3_neighbour"])
                        .agg(F.countDistinct("flight_id").alias("flight_count"),
                            F.collect_list("segment_id").alias("id_list"))
                        .filter(F.col("flight_count") > 1)
                        .drop("flight_count"))
        
            grouped_df = grouped_df.filter(F.size("id_list") > 1)

            # -----------------------------------------------------------------------------
            # Create pairwise combinations using self-join on indexed exploded DataFrame
            # -----------------------------------------------------------------------------
            # Explode id_list to individual rows 
            df_exploded = grouped_df.withColumn("segment_id", explode("id_list")).drop("id_list")
            
            # Add back the flight_level_ft as it will speed up self-joins
            segment_meta_df = resampled_co.select("segment_id", "flight_level_ft", "flight_id")
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
                    (F.abs(F.col("df1.flight_level_ft") - F.col("df2.flight_level_ft")) <= v_dist_ft) &
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
            df_pairs = df_pairs.filter(F.col("ID1") != F.col("ID2")) # should not be necessary as we join on < not <=
            df_pairs = df_pairs.withColumn(
                "ID",
                F.concat_ws("_", F.array_sort(F.array(F.col("ID1"), F.col("ID2"))))
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
            coords_sdf1 = resampled_co.withColumnRenamed("segment_id", "ID1") \
                .withColumnRenamed("latitude", "lat1") \
                .withColumnRenamed("longitude", "lon1") \
                .withColumnRenamed("time_over", "time1") \
                .withColumnRenamed("flight_level_ft", 'altitude_ft1') \
                .withColumnRenamed("flight_id", "flight_id1") \
                .withColumnRenamed("icao24", "icao241") \
                .select("ID1", "lat1", "lon1", "time1", "altitude_ft1", "flight_id1", "icao241")
        
            coords_sdf2 = resampled_co.withColumnRenamed("segment_id", "ID2") \
                .withColumnRenamed("latitude", "lat2") \
                .withColumnRenamed("longitude", "lon2") \
                .withColumnRenamed("time_over", "time2") \
                .withColumnRenamed("flight_level_ft", 'altitude_ft2') \
                .withColumnRenamed("flight_id", "flight_id2") \
                .withColumnRenamed("icao24", "icao242") \
                .select("ID2", "lat2", "lon2", "time2", "altitude_ft2", "flight_id2", "icao242")
        
            coords_sdf1 = coords_sdf1.repartition(p_numb, "ID1")
            coords_sdf2 = coords_sdf2.repartition(p_numb, "ID2")
        
            df_pairs = df_pairs.join(coords_sdf1, on="ID1", how="left")
            df_pairs = df_pairs.join(coords_sdf2, on="ID2", how="left")

            # Round altitude to 0.01 ft precision
            df_pairs = df_pairs\
                .withColumn("altitude_ft1", F.round(F.col("altitude_ft1"), 2)) \
                .withColumn("altitude_ft2", F.round(F.col("altitude_ft2"), 2))

            # -----------------------------------------------------------------------------
            # Calculate and filter based on time differense (s)
            # -----------------------------------------------------------------------------
            df_pairs = df_pairs.withColumn('time_diff_s', F.unix_timestamp(F.col("time1")) - F.unix_timestamp(F.col("time2")))
            df_pairs = df_pairs.filter(F.abs(F.col('time_diff_s')) == 0)
            
            # -----------------------------------------------------------------------------
            # Calculate and filter based on height differense (s)
            # -----------------------------------------------------------------------------
            df_pairs = df_pairs.withColumn('v_dist_ft', F.abs(F.col("altitude_ft1") - F.col("altitude_ft2")))
            df_pairs = df_pairs.filter(F.col('v_dist_ft') <= lit(v_dist_ft))
        
            # -----------------------------------------------------------------------------
            # Calulate and filter based on distance (km)
            # -----------------------------------------------------------------------------
        
            df_pairs = df_pairs.withColumn("lat1_rad", radians(F.col("lat1"))) \
                           .withColumn("lat2_rad", radians(F.col("lat2"))) \
                           .withColumn("lon1_rad", radians(F.col("lon1"))) \
                           .withColumn("lon2_rad", radians(F.col("lon2")))
        
            df_pairs = df_pairs.withColumn(
                "h_dist_NM",
                0.539957 * 2 * CloseEncounters.earth_radius_km * atan2(
                    sqrt(
                        (sin(F.col('lat2_rad') - F.col('lat1_rad')) / 2)**2 +
                        cos(F.col('lat1_rad')) * cos(F.col('lat2_rad')) *
                        (sin(F.col('lon2_rad') - F.col('lon1_rad')) / 2)**2
                    ),
                    sqrt(1 - (
                        (sin(F.col('lat2_rad') - F.col('lat1_rad')) / 2)**2 +
                        cos(F.col('lat1_rad')) * cos(F.col('lat2_rad')) *
                        (sin(F.col('lon2_rad') - F.col('lon1_rad')) / 2)**2
                    ))
                )
            )
        
            df_pairs = df_pairs.drop('lat1_rad', 'lat2_rad', 'lon1_rad', 'lon2_rad')
        
            df_pairs = df_pairs.filter(F.col('h_dist_NM') <= lit(h_dist_NM))
        
            # -----------------------------------------------------------------------------
            # Enrich sample
            # -----------------------------------------------------------------------------

            df_pairs = df_pairs.withColumn(
                'flight_id1',
                F.concat(F.lit('ID_'), F.round(F.col('flight_id1')).cast('int').cast('string'))
            )
            
            df_pairs = df_pairs.withColumn(
                'flight_id2',
                F.concat(F.lit('ID_'), F.round(F.col('flight_id2')).cast('int').cast('string'))
            )

            df_pairs = df_pairs.withColumn(
                "ce_id",
                F.concat_ws(
                    "_",
                    F.sort_array(
                        F.array(F.col("flight_id1"), F.col("flight_id2"))
                    )
                )
            )

            ## Goal: Enrich dataset for Taxonomy
            # Step 0: Cache previous calculations
            df_pairs = df_pairs.cache()
            
            # Step 1: Compute and round distances
            df_pairs = (
                df_pairs
                .withColumn("3D_dist_NM", F.round(F.sqrt((F.col("v_dist_ft") / 6076.12) ** 2 + F.col("h_dist_NM") ** 2), 6))
                .withColumn("v_dist_ft", F.round(F.col("v_dist_ft"), 6))
                .withColumn("h_dist_NM", F.round(F.col("h_dist_NM"), 6))
            )
            
            # Step 2: Create structs to capture values at extrema and boundary times
            df_pairs = df_pairs.withColumns({
                # Start of trajectory (min time_over)
                "start_3D_struct": F.struct(F.col("time_over").alias("time_over"), F.col("3D_dist_NM").alias("3D_dist_NM")),
                "start_v_struct": F.struct(F.col("time_over").alias("time_over"), F.col("v_dist_ft").alias("v_dist_ft")),
                "start_h_struct": F.struct(F.col("time_over").alias("time_over"), F.col("h_dist_NM").alias("h_dist_NM")),
            
                # End of trajectory (max time_over)
                "end_3D_struct": F.struct(F.col("time_over").alias("time_over"), F.col("3D_dist_NM").alias("3D_dist_NM")),
                "end_v_struct": F.struct(F.col("time_over").alias("time_over"), F.col("v_dist_ft").alias("v_dist_ft")),
                "end_h_struct": F.struct(F.col("time_over").alias("time_over"), F.col("h_dist_NM").alias("h_dist_NM")),
            
                # Extremum values
                "min_3D_struct": F.struct(F.col("3D_dist_NM").alias("3D_dist_NM"), F.col("time_over").alias("time_over")),
                "max_3D_struct": F.struct(F.col("3D_dist_NM").alias("3D_dist_NM"), F.col("time_over").alias("time_over")),
            
                "min_v_struct": F.struct(F.col("v_dist_ft").alias("v_dist_ft"), F.col("time_over").alias("time_over")),
                "max_v_struct": F.struct(F.col("v_dist_ft").alias("v_dist_ft"), F.col("time_over").alias("time_over")),
            
                "min_h_struct": F.struct(F.col("h_dist_NM").alias("h_dist_NM"), F.col("time_over").alias("time_over")),
                "max_h_struct": F.struct(F.col("h_dist_NM").alias("h_dist_NM"), F.col("time_over").alias("time_over"))
            })
            
            # Step 3: Group-level aggregation
            agg_df = (
                df_pairs
                .groupBy("ce_id")
                .agg(
                    # Start (min time_over)
                    F.min("start_3D_struct").alias("start_3D_struct"),
                    F.min("start_v_struct").alias("start_v_struct"),
                    F.min("start_h_struct").alias("start_h_struct"),
            
                    # End (max time_over)
                    F.max("end_3D_struct").alias("end_3D_struct"),
                    F.max("end_v_struct").alias("end_v_struct"),
                    F.max("end_h_struct").alias("end_h_struct"),
            
                    # Min/max values
                    F.min("3D_dist_NM").alias("min_3D_dist_NM"),
                    F.max("3D_dist_NM").alias("max_3D_dist_NM"),
                    F.min("v_dist_ft").alias("min_v_dist_ft"),
                    F.max("v_dist_ft").alias("max_v_dist_ft"),
                    F.min("h_dist_NM").alias("min_h_dist_NM"),
                    F.max("h_dist_NM").alias("max_h_dist_NM"),
            
                    # Time of extrema
                    F.min("min_3D_struct").alias("min_3D_struct"),
                    F.max("max_3D_struct").alias("max_3D_struct"),
                    F.min("min_v_struct").alias("min_v_struct"),
                    F.max("max_v_struct").alias("max_v_struct"),
                    F.min("min_h_struct").alias("min_h_struct"),
                    F.max("max_h_struct").alias("max_h_struct")
                )
                .select(
                    "ce_id",
            
                    # Extract start/end times and values
                    F.col("start_3D_struct.time_over").alias("start_time"),
                    F.col("end_3D_struct.time_over").alias("end_time"),
                    F.col("start_3D_struct.3D_dist_NM").alias("start_3D_dist_NM"),
                    F.col("end_3D_struct.3D_dist_NM").alias("end_3D_dist_NM"),
            
                    F.col("start_v_struct.v_dist_ft").alias("start_v_dist_ft"),
                    F.col("end_v_struct.v_dist_ft").alias("end_v_dist_ft"),
            
                    F.col("start_h_struct.h_dist_NM").alias("start_h_dist_NM"),
                    F.col("end_h_struct.h_dist_NM").alias("end_h_dist_NM"),
            
                    # Min/max values
                    "min_3D_dist_NM", "max_3D_dist_NM",
                    "min_v_dist_ft", "max_v_dist_ft",
                    "min_h_dist_NM", "max_h_dist_NM",
            
                    # Times of extrema
                    F.col("min_3D_struct.time_over").alias("min_3D_dist_NM_time"),
                    F.col("max_3D_struct.time_over").alias("max_3D_dist_NM_time"),
                    F.col("min_v_struct.time_over").alias("min_v_dist_ft_time"),
                    F.col("max_v_struct.time_over").alias("max_v_dist_ft_time"),
                    F.col("min_h_struct.time_over").alias("min_h_dist_NM_time"),
                    F.col("max_h_struct.time_over").alias("max_h_dist_NM_time")
                )
            )
            
            # Step 4: Join aggregate results back
            df_pairs = df_pairs.join(agg_df, on="ce_id", how="left")
            
            # Step 5: Add boolean flags
            df_pairs = (
                df_pairs
                .withColumn("is_start_time", F.col("time_over") == F.col("start_time"))
                .withColumn("is_end_time", F.col("time_over") == F.col("end_time"))
                .withColumn("is_min_3D_dist_NM", F.col("3D_dist_NM") == F.col("min_3D_dist_NM"))
                .withColumn("is_max_3D_dist_NM", F.col("3D_dist_NM") == F.col("max_3D_dist_NM"))
                .withColumn("is_min_v_dist_ft", F.col("v_dist_ft") == F.col("min_v_dist_ft"))
                .withColumn("is_max_v_dist_ft", F.col("v_dist_ft") == F.col("max_v_dist_ft"))
                .withColumn("is_min_h_dist_NM", F.col("h_dist_NM") == F.col("min_h_dist_NM"))
                .withColumn("is_max_h_dist_NM", F.col("h_dist_NM") == F.col("max_h_dist_NM"))
            )
            
            # Step 6: Drop intermediate struct columns
            df_pairs = df_pairs.drop(
                "start_3D_struct", "end_3D_struct",
                "start_v_struct", "end_v_struct",
                "start_h_struct", "end_h_struct",
                "min_3D_struct", "max_3D_struct",
                "min_v_struct", "max_v_struct",
                "min_h_struct", "max_h_struct"
            )
            
            # Step 7: Cache final result
            df_pairs.cache()


        if method == 'brute_force':
            # -----------------------------------------------------------------------------
            # Create pairwise combinations using self-join on indexed exploded DataFrame
            # -----------------------------------------------------------------------------
            # Explode id_list to individual rows and add index within each h3 group
            resampled_co = self.resampled_sdf.filter(F.col('flight_level_ft') >= lit(v_cutoff_FL)*100)

            coords_df_f = resampled_co.select(['segment_id','time_over'])
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
            df_pairs = df_pairs.filter(F.col("ID1") != F.col("ID2")) # should not be necessary as we join on < not <=
            df_pairs = df_pairs.withColumn(
                "ID",
                F.concat_ws("_", F.array_sort(F.array(F.col("ID1"), F.col("ID2"))))
            )

            # -----------------------------------------------------------------------------
            # Join with Original Coordinates for Each ID
            # -----------------------------------------------------------------------------
            coords_sdf1 = resampled_co.withColumnRenamed("segment_id", "ID1") \
                .withColumnRenamed("latitude", "lat1") \
                .withColumnRenamed("longitude", "lon1") \
                .withColumnRenamed("time_over", "time1") \
                .withColumnRenamed("flight_level_ft", 'altitude_ft1') \
                .withColumnRenamed("flight_id", "flight_id1") \
                .select("ID1", "lat1", "lon1", "time1", "altitude_ft1", "flight_id1")

            coords_sdf2 = resampled_co.withColumnRenamed("segment_id", "ID2") \
                .withColumnRenamed("latitude", "lat2") \
                .withColumnRenamed("longitude", "lon2") \
                .withColumnRenamed("time_over", "time2") \
                .withColumnRenamed("flight_level_ft", 'altitude_ft2') \
                .withColumnRenamed("flight_id", "flight_id2") \
                .select("ID2", "lat2", "lon2", "time2", "altitude_ft2", "flight_id2")

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
            df_pairs = df_pairs.withColumn('v_dist_ft', F.col("altitude_ft1") - F.col("altitude_ft2"))
            df_pairs = df_pairs.filter(F.abs(F.col('v_dist_ft')) <= lit(v_dist_ft))
            #df_pairs.cache()
            #print(f"Number of pairs after FL filter {df_pairs.count()}")

            # -----------------------------------------------------------------------------
            # Calulate and filter based on distance (km)
            # -----------------------------------------------------------------------------
            df_pairs = df_pairs.withColumn(
                "h_dist_NM",
                0.539957 * 2 * CloseEncounters.earth_radius_km * atan2(
                    sqrt(
                        (sin(radians(F.col("lat2")) - radians(F.col("lat1"))) / 2)**2 +
                        cos(radians(F.col("lat1"))) * cos(radians(F.col("lat2"))) *
                        (sin(radians(F.col("lon2")) - radians(F.col("lon1"))) / 2)**2
                    ),
                    sqrt(1 - (
                        (sin(radians(F.col("lat2")) - radians(F.col("lat1"))) / 2)**2 +
                        cos(radians(F.col("lat1"))) * cos(radians(F.col("lat2"))) *
                        (sin(radians(F.col("lon2")) - radians(F.col("lon1"))) / 2)**2
                    ))
                )
            )

            df_pairs = df_pairs.filter(F.col('h_dist_NM') <= lit(h_dist_NM))

            # # -----------------------------------------------------------------------------
            # # Enrich sample
            # # -----------------------------------------------------------------------------

            df_pairs = df_pairs.withColumn(
                "ce_id",
                F.concat_ws(
                    "_",
                    F.sort_array(
                        F.array(F.col("flight_id1"), F.col("flight_id2"))
                    )
                )
            )

            full_window = (
                Window
                .partitionBy("ce_id")
                .orderBy(F.col("time_over"))
                .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            )

            df_pairs = (
                df_pairs
                # start_time_over = time_over at minimum time_over
                .withColumn(
                    "start_time",
                    F.first(F.col("time_over")).over(full_window)
                )
                # end_time_over = time_over at minimum time_over
                .withColumn(
                    "end_time",
                    F.first(F.col("time_over")).over(full_window)
                )
                # start_v_dist_ft = v_dist_ft at minimum time_over
                .withColumn(
                    "start_v_dist_ft",
                    F.first(F.col("v_dist_ft")).over(full_window)
                )
                # end_v_dist_ft = v_dist_ft at maximum time_over
                .withColumn(
                    "end_v_dist_ft",
                    F.last(F.col("v_dist_ft")).over(full_window)
                )
                # start_h_dist_NM = h_dist_NM at minimum time_over
                .withColumn(
                    "start_h_dist_NM",
                    F.first(F.col("h_dist_NM")).over(full_window)
                )
                # end_h_dist_NM = h_dist_NM at maximum time_over
                .withColumn(
                    "end_h_dist_NM",
                    F.last(F.col("h_dist_NM")).over(full_window)
                )
                # min_v_dist_ft = min(v_dist_ft) at any time_over
                .withColumn(
                    "min_v_dist_ft",
                    F.min(F.col("v_dist_ft")).over(full_window)
                )
                # max_v_dist_ft = max(v_dist_ft) at any time_over
                .withColumn(
                    "max_v_dist_ft",
                    F.max(F.col("v_dist_ft")).over(full_window)
                )
                # min_h_dist_NM = min(h_dist_NM) at any time_over
                .withColumn(
                    "min_h_dist_NM",
                    F.min(F.col("h_dist_NM")).over(full_window)
                )
                # max_v_dist_ft = max(v_dist_ft) at any time_over
                .withColumn(
                    "max_h_dist_NM",
                    F.max(F.col("h_dist_NM")).over(full_window)
                )
            )

            df_pairs.cache()
 
        logger.info("Found %d candidate close encounters", df_pairs.count())
        
        self.close_encounter_sdf = df_pairs
        self.close_encounter_pdf = df_pairs.toPandas()

        if output == 'pdf':
            return self.close_encounter_pdf
        
        if output == 'sdf':
            return self.close_encounter_sdf

        if output is None:
            return self
            
    def create_keplergl_map(self, display: bool = False) -> KeplerGl:
        """
        Create a Kepler.gl map visualizing detected close encounters.
    
        Args:
            display (bool, optional): If True, display the map inline in a notebook. Defaults to False.
    
        Returns:
            KeplerGl: Kepler.gl map object ready for display or saving.
    
        Raises:
            ValueError: If no encounter data is available.
        """
        if self.close_encounter_pdf is None:
            raise ValueError(logger.error("Kepler map creation failed: no encounter data found."))
    
        encounters_df = self.close_encounter_pdf
        keplergl_config = self._load_kepler_config()
        data_id = 'Close Encounter Data'
    
        # Update config with encounter data ranges
        keplergl_config['config']['visState']['filters'][0]['value'] = [
            encounters_df.time_over.min().timestamp() * 1000,
            encounters_df.time_over.max().timestamp() * 1000
        ]
        keplergl_config['config']['visState']['interactionConfig']['tooltip']['fieldsToShow'][data_id][1]['filterProps']['domain'] = encounters_df['flight_id1'].to_list()
        keplergl_config['config']['visState']['interactionConfig']['tooltip']['fieldsToShow'][data_id][2]['filterProps']['domain'] = encounters_df['flight_id2'].to_list()
    
        kepler_map = KeplerGl(data={data_id: encounters_df}, config=keplergl_config)
    
        if display:
            try:
                kepler_map.show()
            except Exception:
                print("Inline map display failed — are you in a compatible notebook environment?")
    
        return kepler_map


    def save_keplergl_map(self, kepler_map: KeplerGl, filename: str = None) -> None:
        """
        Save a Kepler.gl map object to an HTML file.
    
        Args:
            kepler_map (KeplerGl): The Kepler.gl map to save.
            filename (str, optional): Output file name. Defaults to a timestamped name.
    
        Returns:
            None
        """
        if filename is None:
            filename = f'close_encounters_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
    
        kepler_map.save_to_html(file_name=filename)
        
        logger.info(f"Kepler map saved to: {filename}. Note: To view it, (download and) open with a browser.")
    
    def get_close_encounters_sdf(self) -> DataFrame:
        """
        Return the close encounter results as a Spark DataFrame.
    
        Returns:
            DataFrame: Close encounter pairs with spatial and temporal data.
    
        Raises:
            ValueError: If results have not been computed yet.
        """
        
        if self.close_encounter_sdf is None:
            raise ValueError("No close encounter data found. Run find_close_encounters() or set manually.")
        return self.close_encounter_sdf
    
    def get_close_encounters_pdf(self) -> pd.DataFrame:
        """
        Return the close encounter results as a pandas DataFrame.
    
        Returns:
            pd.DataFrame: Close encounter pairs with spatial and temporal data.
    
        Raises:
            ValueError: If results have not been computed yet.
        """
        if self.close_encounter_pdf is None:
            raise ValueError("No close encounter data found. Run find_close_encounters() or set manually.")
        return self.close_encounter_pdf

    def get_resampled_trajectories_sdf(self) -> DataFrame:
        """
        Return the resampled Spark DataFrame of trajectories.
    
        Returns:
            DataFrame: Resampled trajectory data.
    
        Raises:
            ValueError: If resampling has not been performed yet.
        """
        if self.resampled_sdf is None:
            raise ValueError("Resampled trajectories not found. Run resample() first.")
        return self.resampled_sdf

    def get_raw_trajectories_sdf(self) -> DataFrame:
        """
        Return the raw Spark DataFrame of loaded trajectories.
    
        Returns:
            DataFrame: Original input trajectory data.
    
        Raises:
            ValueError: If no trajectory data is loaded.
        """
        if self.traj_sdf is None:
            raise ValueError("No trajectory data loaded. Use load_*_trajectories() methods first.")
        return self.traj_sdf

    def get_summary(self) -> dict:
        """
        Return a dictionary summarizing the current state of the class.
    
        Returns:
            dict: Summary statistics and current configuration.
        """
        summary = {
            "flights_loaded": None,
            "segments_resampled": None,
            "close_encounters_found": None,
            "resample_freq_s": self.resample_freq_s,
            "resample_t_max": self.resample_t_max
        }
    
        if self.traj_sdf is not None:
            summary["flights_loaded"] = self.traj_sdf.select("flight_id").distinct().count()
    
        if self.resampled_sdf is not None:
            summary["segments_resampled"] = self.resampled_sdf.count()
    
        if self.close_encounter_sdf is not None:
            summary["close_encounters_found"] = self.close_encounter_sdf.count()
        
        logger.info("Generated analysis summary: %s", summary)
        return summary

    def set_close_encounter_pdf(self, pdf: pd.DataFrame):
        """
        Set the close encounter results from a DataFrame or Parquet file.
    
        Args:
            pdf (pd.DataFrame): Pandas DataFrame of results or path to a Parquet file.
    
        Raises:
            TypeError: If input is neither a DataFrame nor a string path.
        """
        if isinstance(pdf, pd.DataFrame):
            self.close_encounter_pdf = pdf
        elif isinstance(pdf, str):
            self.close_encounter_pdf = pd.read_parquet(pdf)
            logger.info("Set close encounter results from %s", "DataFrame" if isinstance(pdf, pd.DataFrame) else pdf)
        else:
            raise TypeError("Input must be a pandas DataFrame or a string path to a Parquet file.")
        return self

    def check_trajectories_loaded(self) -> bool:
        """
        Check whether trajectory data has been successfully loaded.
    
        Returns:
            bool: True if loaded, False otherwise.
        """
        return self.traj_sdf is not None

    def check_trajectories_resampled(self) -> bool:
        """
        Check whether resampling has been performed.
    
        Returns:
            bool: True if resampled trajectories are available, False otherwise.
        """
        return self.resampled_sdf is not None

    def check_ce_present(self) -> bool:
        """
        Check whether resampling has been performed.
    
        Returns:
            bool: True if resampled trajectories are available, False otherwise.
        """
        return self.close_encounters_sdf is not None

    def filter_down_ce(self) -> pd.DataFrame:
        """
        Apply filtering rules to encounters
        """

        # Rule R1: Exclude encounters for which 
        #       flight_id1 == flight_id2
        #                  OR
        #       icao241 == icao242
        close_encounter_sdf = self.close_encounter_sdf
        close_encounter_sdf =  close_encounter_sdf\
            .filter(F.col('flight_id1')!=col('flight_id2'))\
            .filter(F.col('icao241')!=col('icao242'))
        
        # Rule R2: Unclear
        
        # Rule R3
        # For each pair, check duration <10 sec and increasing horizontal/vertical separation after the minimum 

        close_encounter_sdf = close_encounter_sdf.filter(
            ~(
               (F.col('start_v_dist_ft') <= F.col('end_v_dist_ft')) &
               (F.col('start_h_dist_NM') <= F.col('end_h_dist_NM')) &
                (
                    F.abs(
                        F.unix_timestamp('end_time')
                        - F.unix_timestamp('start_time')
                    ) <= 10
                )
            )
        )

    def find_close_encounters_duckdb(
        self,
        h_dist_NM=5,
        v_dist_ft=1000,
        v_cutoff_FL=245,
        freq_s=5,
        t_max=10,
        p_numb=100
         ):
            """
            DuckDB-based detection that exactly reproduces the Spark half-disk logic,
            including the explicit group/collect_list/filter stage.
            """
            if self.traj_sdf is None:
                raise ValueError("No trajectories loaded.")
            logger.info("Starting DuckDB-based detection (1:1 with Spark).")

            # 1) resample + altitude cutoff
            self.resample(freq_s=freq_s, t_max=t_max, p_numb=p_numb)
            resolution = self._select_resolution_half_disk(h_dist_NM=h_dist_NM)

            base_sdf = (
                self.resampled_sdf
                    .filter(F.col("flight_level_ft") >= v_cutoff_FL * 100)
                    .select("segment_id","flight_id","icao24","latitude","longitude","time_over","flight_level_ft")
            )

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "sample.parquet")
                base_sdf.write.mode("overwrite").parquet(path)

                con = duckdb.connect()
                con.execute("INSTALL h3 FROM community; LOAD h3;")
                con.execute("PRAGMA max_temp_directory_size='500GiB';")
                con.execute("SET preserve_insertion_order = false;")
                con.execute("SET threads = 4;")

                # A) load & compute h3 index
                con.execute(f"""
                CREATE TABLE resampled AS
                SELECT
                *,
                h3_latlng_to_cell_string(latitude, longitude, {resolution}) AS h3_index
                FROM parquet_scan('{path}/*.parquet');
                """)

                # B) explode into neighbors
                con.execute("""
                CREATE TABLE exploded AS
                SELECT
                *,
                unnest(array_concat([h3_index], h3_grid_ring(h3_index, 1))) AS h3_neighbour
                FROM resampled;
                """)

                # C) now do the exact Spark group/collect_list/filter
                con.execute("""
                CREATE TABLE grouped AS
                SELECT
                time_over,
                h3_neighbour,
                COUNT(DISTINCT flight_id) AS flight_count,
                LIST(segment_id)         AS id_list
                FROM exploded
                GROUP BY time_over, h3_neighbour
                HAVING flight_count > 1;
                """)

                # D) explode that list back out to get one row per segment:
                con.execute("""
                CREATE TABLE exploded2 AS
                SELECT
                time_over,
                h3_neighbour,
                unnest(id_list) AS segment_id
                FROM grouped;
                """)

                # E) re-join the metadata (still only the filtered segments)
                con.execute("""
                CREATE TABLE enriched AS
                SELECT
                e2.*,
                r.flight_level_ft
                FROM exploded2 e2
                JOIN resampled r
                ON e2.segment_id = r.segment_id
                AND e2.time_over   = r.time_over;
                """)

                # F) index them per (time,h3) for the self-join
                con.execute("""
                CREATE TABLE indexed AS
                SELECT
                *,
                row_number() OVER (
                    PARTITION BY time_over, h3_neighbour
                    ORDER BY segment_id
                ) AS idx
                FROM enriched;
                """)

                # G) self-join exactly as Spark does (including vertical filter in join)
                con.execute(f"""
                CREATE TABLE pairs AS
                SELECT DISTINCT
                a.time_over,
                a.segment_id   AS ID1,
                b.segment_id   AS ID2
                FROM indexed a
                JOIN indexed b
                ON a.time_over     = b.time_over
                AND a.h3_neighbour = b.h3_neighbour
                AND a.idx < b.idx
                AND ABS(ROUND(a.flight_level_ft, 2) - ROUND(b.flight_level_ft, 2)) <= {v_dist_ft};
                """)        

                # H) stitch back coordinates for each side (again only from the filtered set)
                con.execute("""
                CREATE TABLE joined AS
                SELECT
                p.*,
                r1.latitude        AS lat1,
                r1.longitude       AS lon1,
                ROUND(r1.flight_level_ft,2) AS altitude_ft1,
                r1.flight_id       AS flight_id1,
                r2.latitude        AS lat2,
                r2.longitude       AS lon2,
                ROUND(r2.flight_level_ft, 2) AS altitude_ft2,
                r2.flight_id       AS flight_id2
                FROM pairs p
                JOIN resampled r1
                ON p.ID1 = r1.segment_id
                AND p.time_over = r1.time_over
                JOIN resampled r2
                ON p.ID2 = r2.segment_id
                AND p.time_over = r2.time_over;
                """)

                # I) final horizontal-distance filter
                con.execute(f"""
                SELECT *,
                0.539957 * 2 * {self.earth_radius_km} * atan2(
                    sqrt(
                    pow(sin(radians(lat2-lat1)/2),2) +
                    cos(radians(lat1))*cos(radians(lat2)) *
                    pow(sin(radians(lon2-lon1)/2),2)
                    ),
                    sqrt(1 - (
                    pow(sin(radians(lat2-lat1)/2),2) +
                    cos(radians(lat1))*cos(radians(lat2)) *
                    pow(sin(radians(lon2-lon1)/2),2)
                    ))
                ) AS h_dist_NM
                FROM joined
                WHERE h_dist_NM <= {h_dist_NM};
                """)

                df = con.df()
                logger.info("DuckDB (1:1 Spark) found %d encounters", len(df))
                self.close_encounter_pdf = df
                return df

    def expand_close_encounters(
        self,
        pre_seconds: int = 300,
        post_seconds: int = 300,
        step_seconds: int = 1
    ):
        """
        Expands the limited trajectories in the close_encounters_sdf before and after the close encounters.
        
        :param df:          original Spark DataFrame
        :param pre_seconds: seconds before start_time col to begin the fill window
        :param post_seconds:seconds after  end_time col to end the fill window
        :param step_seconds:interval step for the sequence, in seconds
        :return:            a new DataFrame with one row per second in the
                            extended window, original data where available,
                            NULLs elsewhere
        """
        logger.info("Starting close encounter expansion with method='%s'", method)
        
        # Check if data is there
        if (~self.check_ce_present() or ~self.check_trajectories_resampled()):
            raise Exception("No trajectories resampled OR close encounters not yet calculated. \
            Use .resample() and/or .find_close_encounters() before running this.")

        # Define nested helper function
        def add_col_suffix(
            sdf : DataFrame, 
            suffix: str
        ) -> DataFrame:
            """
            Rename each column in the Spark DataFrame sdf by adding the suffix at the end.
    
            :param sdf:         original Spark DataFrame
            :param suffix:      string to be added at the end
            :return:            a new DataFrame with renamed columns (now containing suffix at the end of each column name).
            """
            renamed = []
            for c in sdf.columns:
                renamed.append(F.col(c).alias(f"{c}{suffix}"))
            return sdf.select(*renamed)
        
        def add_empty_rows(
            df: DataFrame,
            pre_seconds: int = 300,
            post_seconds: int = 300,
            step_seconds: int = 1
        ) -> DataFrame:
            """
            For each distinct id_col='ce_id' in df, generate a timestamp sequence from
            (start_col='start_time' - pre_seconds=) to (end_col='end_time' + post_seconds) at step_seconds
            resolution, then left-join to df so that every second in that full
            range appears — with original values where they exist and NULLs elsewhere.
        
            :param df:          original Spark DataFrame
            :param pre_seconds: seconds before start_time col to begin the fill window
            :param post_seconds:seconds after  end_time col to end   the fill window
            :param step_seconds:interval step for the sequence, in seconds
            :return:            a new DataFrame with one row per second in the
                                extended window, original data where available,
                                NULLs elsewhere
            """

            # Column names definitions
            id_col="ce_id"
            time_col="time_over"
            start_col="start_time"
            end_col="end_time"
            flight_id1_col='flight_id1'
            flight_id2_col='flight_id2'
            
            #
            
            # 1. Extract each (id, start, end) window
            windows = (
                df
                .select(id_col, start_col, end_col, flight_id1_col, flight_id2_col)
                .distinct()
            )
        
            # 2. Build a sequence of timestamps for each id
            windows = windows.withColumn(
                "_ts_array",
                sequence(
                    F.expr(f"{start_col} - interval {pre_seconds} seconds"),
                    F.expr(f"{end_col} + interval {post_seconds} seconds"),
                    F.expr(f"interval {step_seconds} seconds")
                )
            )
        
            # 3. Explode into one-row-per-timestamp
            windows = (
                windows
                .withColumn(time_col, explode(col("_ts_array")))
                .select(id_col, time_col, flight_id1_col, flight_id2_col)
            )
        
            # 4. Left-join back to the full sequence so missing fields become NULL
            result = (
                windows
                .join(df, on=[id_col, flight_id1_col, flight_id2_col, time_col], how="left")
                .orderBy(id_col, time_col)
            )
        
            return result

        # Get data
        resampled_sdf = self.get_resampled_sdf()
        ce = self.get_close_encounters_sdf()

        ce_expanded = add_empty_rows(
            df=ce,
            pre_seconds=600,
            post_seconds=600,
            step_seconds=1
        )

        resampled_sdf1 = add_col_suffix(resampled_sdf, suffix='1')\
            .withColumn('flight_id1', col('flight_id1').cast(StringType()))
        
        resampled_sdf2 = add_col_suffix(resampled_sdf, suffix='2')\
            .withColumn('flight_id2', col('flight_id2').cast(StringType()))


        ce_expanded = ce_expanded.join(resampled_sdf1,
           (resampled_sdf1.flight_id1 ==  ce_expanded.flight_id1) &
           (resampled_sdf1.time_over1 == ce_expanded.time_over),
           "left").join(resampled_sdf2,
           (resampled_sdf2.flight_id2 ==  ce_expanded.flight_id2) &
           (resampled_sdf2.time_over2 == ce_expanded.time_over),
           "left")

        return ce_expanded.toPandas()
        