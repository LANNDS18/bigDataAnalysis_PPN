"""
COMP336 Assignment 2 Part 1
Wuwei ZHANG
201522671
"""

import datetime
import geopy.distance
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.window import Window
from pyspark.sql.functions import struct, udf, col, desc, row_number, min, max, lag
from pyspark.sql.types import *

sc = SparkContext(master="local[*]")
spark = SparkSession.builder.appName("Assignment 2 Wuwei Zhang").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.getConf().getAll()

schema = StructType([
    StructField("UserId", IntegerType(), True),
    StructField("Latitude", DoubleType(), True),
    StructField("Longitude", DoubleType(), True),
    StructField("AllZero", IntegerType(), True),
    StructField("Altitude", DoubleType(), True),
    StructField("Timestamp", DoubleType(), True),
    StructField("Date", StringType(), True),
    StructField("Time", StringType(), True),
])

df = spark.read.csv('dataset.txt', header=True, schema=schema)
# Timestamp is sufficient to re-calculate the date and time
df = df.drop('Date', 'Time')
df.printSchema()

# Time difference between Beijing and UTC
time_diff = 8 * 3600
# the difference between Epoch & Unix Timestamp and current time stamp base
interval = (datetime.datetime(1970, 1, 1) - datetime.datetime(1899, 12, 30)).total_seconds()
# change the timestamp to Unix Timestamp + 8 hours
df = df.withColumn('Timestamp', col('Timestamp') * 24 * 3600 + time_diff - interval)

to_date = udf(lambda x: datetime.datetime.fromtimestamp(x.Timestamp, datetime.timezone.utc).strftime('%Y-%m-%d'),
              StringType())
to_time = udf(lambda x: datetime.datetime.fromtimestamp(x.Timestamp, datetime.timezone.utc).strftime('%H:%M:%S'),
              StringType())

df = df.withColumn("Date", to_date(struct([df[x] for x in df.columns])))
df = df.withColumn("Time", to_time(struct([df[x] for x in df.columns])))

print('Step 1: Shift the time to beijing time')
# Combine the latitude and longitude to a single column
df = df.withColumn("Coordinate", struct(df.Latitude, df.Longitude)).drop('Latitude', 'Longitude')
df.show(5)

'''
Calculate for each person, on how many days was the data recorded for them (count any day with
at least one data point).
'''
print('Step 2: Calculate for each person, on how many days was the data recorded for them (count any day with at '
      'least one data point)')

date_df = df.select('UserId', 'Date').drop_duplicates()
date_count = date_df.groupBy('UserID').count().sort(desc('count'))
date_count.show(5)

'''
Calculate for each person, on how many days there were more than 100 data points recorded for them (count any day with at least 100 data points). Output all user IDs and the corresponding value of this measure.
'''
print('Step 3: Calculate for each person, on how many days there were more than 100 data points recorded for them ('
      'count any day with at least 100 data points). Output all user IDs and the corresponding value of this measure.')

date_duplicate = df.select('UserId', 'Date').groupBy('UserID', 'Date').count().filter(col('count') >= 100)
df_user_date = date_duplicate.select('UserId').groupBy('UserID').count().sort(desc('count'))
df_user_date.show(df_user_date.count())

'''
Calculate for each person, the highest altitude that they reached. Output the top 5 user ID according to this measure, its value and the day that was achieved.
'''
print('Step 4: Calculate for each person, the highest altitude that they reached. Output the top 5 user ID according '
      'to this measure, its value and the day that was achieved.')

windowDept = Window.partitionBy("UserId").orderBy(col("Altitude").desc())
df_altitude = df.select('UserId', 'Altitude', 'Date')
df_altitude = df_altitude.withColumn("row", row_number().over(windowDept)).filter(col("row") == 1).drop("row").sort(
    desc('Altitude'))
df_altitude.show(5)

'''
Calculate for each person, the timespan of the observation, i.e., the difference between the highest timestamp of his/her observation and the lowest one. Output the top 5 user ID according to this measure and its value.
'''
print('Step 5: Calculate for each person, the timespan of the observation, i.e., the difference between the highest '
      'timestamp of his/her observation and the lowest one. Output the top 5 user ID according to this measure and '
      'its value.')

df_timestamp_extrema = df.select(
    'UserId', 'Timestamp',
).groupBy(
    'UserId'
).agg(
    min('Timestamp').alias('Timestamp_min'),
    max('Timestamp').alias('Timestamp_max'),
)

time_diff = udf(lambda x: x.Timestamp_max - x.Timestamp_min, DoubleType())
df_timespan = df_timestamp_extrema.withColumn('TimeSpan', time_diff(
    struct([df_timestamp_extrema[x] for x in df_timestamp_extrema.columns]))).sort(desc('TimeSpan'))
df_timespan = df_timespan.drop('Timestamp_min', 'Timestamp_max')
df_timespan.show(5)

'''
Calculate for each person, the distance travelled by them each day. For each user output the (earliest) day they travelled the most. Also, output the total distance travelled by all users on all days.
'''
print('Step 6: Calculate for each person, the distance travelled by them each day. For each user output the ('
      'earliest) day they travelled the most. Also, output the total distance travelled by all users on all days.')

df_distance = df.select('UserId', 'Coordinate', 'Date', 'Timestamp')


# calculate the distance between two points
def get_distance_between_two_points(row):
    coord1 = row.Coordinate
    coord2 = row.lag_Coordinate
    if coord1 is None or coord2 is None:
        return 0
    else:
        return geopy.distance.geodesic(coord1, coord2).km


# lag the coordinate column
same_day_window = Window.partitionBy("UserID", "Date").orderBy("Timestamp")
df_distance = df_distance.withColumn('lag_Coordinate', lag(df_distance['Coordinate']).over(same_day_window))

distance_udf = udf(get_distance_between_two_points, DoubleType())
# calculate the distance travelled by each user each coordinate change
df_distance = df_distance.withColumn('DistanceByCoordinate',
                                     distance_udf(struct([df_distance[x] for x in df_distance.columns])))
# calculate the total distance travelled by each user each day
df_distance_agg = df_distance.groupBy('UserID', 'Date').sum('DistanceByCoordinate').sort(
    desc('sum(DistanceByCoordinate)')).withColumnRenamed('sum(DistanceByCoordinate)', "DistanceByDay")

# calculate the max daily distance travelled by each user
distance_user_window = Window.partitionBy("UserID").orderBy(col('DistanceByDay').desc())
df_distance_user = df_distance_agg.select('UserId', 'DistanceByDay', 'Date').withColumn("row", row_number().over(
    distance_user_window)).filter(col("row") == 1).drop("row").sort(desc('DistanceByDay'))


df_distance_user.sort('UserId').show(df_distance_user.count())

print('The total sum of the total distance for all users:')

df_distance.select('DistanceByCoordinate').groupBy().sum('DistanceByCoordinate').show()
