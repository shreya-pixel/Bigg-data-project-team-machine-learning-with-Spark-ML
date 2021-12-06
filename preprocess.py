from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
#from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Tokenizer 
from pyspark.ml.feature import HashingTF, IDF
import pandas as pd
import time 
count = 1
while True:
    file = "train_batch_"+str(count) + ".csv"
    print(file)
    trainData_df = spark.read.csv(file ,inferSchema = True)
    #trainData_df.mapPartitions(iterator => iterator.drop(1))
    #trainData_df = trainData_df.tail(trainData_df.shape[0] -1)
    trainData_df = trainData_df.select(col("_c0").alias("sentimentcol"),col("_c1").alias("tweetcol"))
    trainData_df.show()
    tweet_token = Tokenizer(inputCol="tweetcol",outputCol="words")
    tweet_tokenized_df = tweet_token.transform(trainData_df)
    #tweet_tokenized_df.show()
    hashingtweetTF = HashingTF(inputCol = "words",outputCol = "rawFeatures",numFeatures = 20)
    tweet_hfTF_df = hashingtweetTF.transform(tweet_tokenized_df)
    #tweet_hfTF_df.show()
    tweet_idf = IDF(inputCol = "rawFeatures",outputCol = "tweet_idf_features")
    tweet_idfModel = tweet_idf.fit(tweet_hfTF_df)
    tweet_tfidf = tweet_idfModel.transform(tweet_hfTF_df)
    tweet_tfidf.show()
    #tweet_tfidf.to_csv('preprocess.csv', mode='a', index=False, header=False)
    #tweet_tfidf.write.save(path='preprocess.csv', format='csv', mode='append', sep='	')
    #tweet_tfidf.write.csv(path="preprocess.csv", mode="append")
    #tweet_tfidf.to_csv("train_batch_"+str(count) + ".csv", index=False, encoding='utf-8-sig')     
    #data = pd.read_csv("preprocess.csv") 
    #print(data)
    count += 1
    time.sleep(2)





