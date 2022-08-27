from collections import OrderedDict

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.types import Row
from pyspark.sql.functions import col,dayofmonth,year,second,minute
from pyspark.sql.functions import *
from pyspark.ml.functions import vector_to_array
import json
import re
import pandas as pd
from pyspark.ml.feature import VectorAssembler,MinMaxScaler,StringIndexer,FeatureHasher
from sklearn.preprocessing import LabelEncoder
from itertools import chain
import matplotlib.pyplot as plt

data_dict = {'FRAUD':1, 'SUICIDE':2, 'SEX OFFENSES FORCIBLE':3, 'LIQUOR LAWS':4, 
'SECONDARY CODES':5, 'FAMILY OFFENSES':6, 'MISSING PERSON':7, 'OTHER OFFENSES':8, 
'DRIVING UNDER THE INFLUENCE':9, 'WARRANTS':10, 'ARSON':11, 'SEX OFFENSES NON FORCIBLE':12,
'FORGERY/COUNTERFEITING':13, 'GAMBLING':14, 'BRIBERY':15, 'ASSAULT':16, 'DRUNKENNESS':17,
'EXTORTION':18, 'TREA':19, 'WEAPON LAWS':20, 'LOITERING':21, 'SUSPICIOUS OCC':22, 
'ROBBERY':23, 'PROSTITUTION':24, 'EMBEZZLEMENT':25, 'BAD CHECKS':26, 'DISORDERLY CONDUCT':27,
'RUNAWAY':28, 'RECOVERED VEHICLE':29, 'VANDALISM':30,'DRUG/NARCOTIC':31, 
'PORNOGRAPHY/OBSCENE MAT':32, 'TRESPASS':33,'VEHICLE THEFT':34, 'NON-CRIMINAL':35, 
'STOLEN PROPERTY':36, 'LARCENY/THEFT':37, 'KIDNAPPING':38,'BURGLARY':39}


pd_dis={'MISSION':1,'BAYVIEW':2,'CENTRAL':3,'TARAVAL':4, 'TENDERLOIN':5,'INGLESIDE':6, 'PARK':7,'SOUTHERN':8, 'RICHMOND':9,'NORTHERN':10}

def preprocess(data_df):

    #Combining columns X and Y
    vector_assembler=VectorAssembler(inputCols=['X','Y'],outputCol="Co-Ordinates")
    combined_XY=vector_assembler.transform(data_df)
    
    #Dropping redundant columns
    cols=['X','Y']
    combined_XY=combined_XY.drop(*cols)
    #combined_XY.show(5)
    
    #Splitting the date
    combined_split_date=combined_XY.withColumn("day",dayofmonth(col("Dates")))\
                                    .withColumn("month",date_format(col("Dates"),"MM"))\
                                    .withColumn("year",year(col("Dates")))\
                                    .withColumn("second",second(col("Dates")))\
                                    .withColumn("minute",minute(col("Dates")))\
                                    .withColumn("hour",hour(col("Dates")))
    combined_split_date.withColumn("month",combined_split_date["month"].cast(IntegerType()))        #Casting the month column to IntegerType()
    
    #print(type(combined_split_date))
    #combined_split_date.show(5)
    #combined_split_date.printSchema()
    
    #Scaling the Co-Ordinates column
    minmax_scaler=MinMaxScaler(inputCol="Co-Ordinates",outputCol="Scaled_XY")
    scaled=minmax_scaler.fit(combined_split_date).transform(combined_split_date)
    #scaled.show(5)
    
    #Converting Category column to numerical values
    map_expr1=create_map([lit(x) for x in chain(*data_dict.items())])
    map_expr2=create_map([lit(x) for x in chain(*pd_dis.items())])
    cat_to_num=scaled.withColumn("label",map_expr1.getItem(col("Category")))
    cat_to_num=cat_to_num.withColumn("pd_district",map_expr2.getItem(col("PdDistrict")))
    
    #Getting only the required columns
    final_df=cat_to_num.select('pd_district','hour','minute','year','Scaled_XY','label')
    final_df=final_df.withColumn('s',vector_to_array('Scaled_XY')).select(['pd_district','hour','minute','year','label']+[col('s')[i] for i in range(2)])
    
    #Featurizing required columns
    feature_vector=FeatureHasher(inputCols=['pd_district','hour','minute','year','s[0]','s[1]'],outputCol="Features")
    featurized_vector=feature_vector.transform(final_df)
    featurized_vector.show(5)
    
    #return featurized_vector