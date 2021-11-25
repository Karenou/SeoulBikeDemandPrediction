from pyspark.sql import SQLContext
from pyspark.sql import functions as func
from pyspark.sql.types import *


def read_data(spark, prop):
    db_name = "big_data_5003"
    tb_name = "SeoulBike"
    url = 'jdbc:mysql://127.0.0.1:3306/%s' % db_name
    df = spark.read.jdbc(url=url, table=tb_name, properties=prop)
    return df

@func.udf(StringType())
def getWeekday(x):
    dayofweek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return dayofweek[int(x)-1]

def feature_engineering(df):
    # extract month, weekday, weekend 
    df = df.withColumn("Date", func.to_date("Date", "dd/MM/yyyy"))
    df = df.withColumn("Month", func.month("Date"))
    df = df.withColumn("Weekday", func.dayofweek("Date")) \
           .withColumn("Weekday", getWeekday("Weekday")) \
           .withColumn("Weekend", 
                    func.udf(lambda x: 1 if x in ["Saturday", "Sunday"] else 0, IntegerType())("Weekday"))
    
    # perform one-hot on categorical variables
    weekday = df.groupBy("id") \
                .pivot("Weekday") \
                .agg(func.count("Weekday")) \
                .fillna(0)

    seasons = df.groupBy("id") \
                .pivot("Seasons")\
                .agg(func.count("Seasons")) \
                .fillna(0)

    holiday = df.groupBy("id") \
                .pivot("Holiday")\
                .agg(func.count("Holiday")) \
                .fillna(0) \
                .withColumnRenamed("Holiday", "has_holiday")
    
    df = df.join(weekday, on="id", how="inner") \
           .join(seasons, on="id", how="inner") \
           .join(holiday.select("id", "has_holiday"), on="id", how="inner")

    df = df.withColumn("FunctioningDay", 
                   func.udf(lambda x: 1 if x == "Yes" else 0, IntegerType())("FunctioningDay"))

    feature_col = [c for c in df.columns if c not in 
               ["RentedBikeCount", "id", "Date", "Seasons", "Holiday","Weekday"]
              ]
        
    return feature_col, df
