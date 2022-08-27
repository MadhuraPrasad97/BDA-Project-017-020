from collections import OrderedDict

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.types import Row
from pyspark.sql.functions import col,dayofmonth,year,second,minute
from pyspark.sql.functions import *
import json
import re
import pandas as pd
from pyspark.ml.feature import VectorAssembler,MinMaxScaler,StringIndexer
from sklearn.preprocessing import LabelEncoder
import preprocess

sc = SparkContext("local[2]", "BDA")
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 1)
spark = SparkSession.builder.getOrCreate()

def temp(rdd):
    df = spark.read.json(rdd)

    columns = ["Dates", "Category", "Description", "DayOfWeek", "PdDistrict", "Resolution", "Address", "X", "Y"]

    for rows in df.rdd.toLocalIterator():
        for row in rows:
            l = eval(row)
            data = []

            for line in l:
                line = re.sub('\n', '', line)  # To remove the \n character
                line = re.sub(r',(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', "",
                              line)  # To remove comma which is not the delimiter
                line = re.sub('"', '', line)  # To remove the enclosure(Double quotes)

                if not 'X' in line.split(','):  # To skip the header line
                    data.append(line.split(','))

            conv_df = spark.createDataFrame(data, columns)
            data.clear()
            #conv_df = conv_df.toPandas()
            # conv_df.show(5)
            #print(type(conv_df))
            
            conv_df=conv_df.withColumn("Dates",col("Dates").cast(TimestampType()))\
                    .withColumn("Category",col("Category").cast(StringType()))\
                    .withColumn("Description",col("Description").cast(StringType()))\
                    .withColumn("DayOfWeek",col("DayOfWeek").cast(StringType()))\
                    .withColumn("PdDistrict",col("PdDistrict").cast(StringType()))\
                    .withColumn("Resolution",col("Resolution").cast(StringType()))\
                    .withColumn("Address",col("Address").cast(StringType()))\
                    .withColumn("X",col("X").cast(FloatType()))\
                    .withColumn("Y",col("Y").cast(FloatType()))
            processed_df=preprocess(conv_df)
            #conv_df.printSchema()

lines = ssc.socketTextStream('localhost', 6100)

lines.foreachRDD(lambda rdd: temp(rdd))
ssc.start()
ssc.awaitTermination()