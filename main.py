from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as func
from preprocessing import read_data, feature_engineering
from linear_regression import linear_regression
from tree import random_forest, gradient_boosting_tree
import pandas as pd

if __name__ == "__main__":

    spark = SparkSession.builder \
                        .master("local") \
                        .appName("RentBikeCountPrediction") \
                        .getOrCreate()
 
    prop = {'user': 'root',
        'password': '！Bigdata5003',
        'driver': 'com.mysql.cj.jdbc.Driver'}

    # load data and perform feature engineering
    data = read_data(spark, prop)
    data.cache()
    feature_col, data = feature_engineering(data)

    label_col = "RentedBikeCount"

    # create feature vector
    feat_assembler = VectorAssembler(inputCols=feature_col, outputCol='features')
    model_df = feat_assembler.transform(data)
    train_df, val_df, test_df = model_df.select("features", label_col) \
                                        .randomSplit([0.6, 0.2, 0.2], seed=100)

    lr_params = {
        "maxIter": [5, 10, 15], 
        "regParam": [0.1, 0.3]  
    }

    lr_best_param, lr_model = linear_regression(train_df, val_df, test_df, label_col, lr_params)
    lr_model.save("./model/lr.model")

    tree_params = {
        "numTrees": [100, 120, 150],
        "maxDepth": [4, 8, 12],
        "subsamplingRate": [1.0],
        "featureSubsetStrategy": ["all"],
        "minInstancesPerNode": [1]
    }

    rf_best_param, rf_model = random_forest(train_df, val_df, test_df, label_col, tree_params)
    rf_model.save("./model/rf.model")

    gbdt_best_param, gbdt_model = gradient_boosting_tree(train_df, val_df, test_df, label_col, tree_params)
    gbdt_model.save("./model/gbdt.model")

    # generate prediction by ensembling rf and gbdt
    rf_pred = rf_model.transform(model_df.select("features", label_col))
    rf_pred = rf_pred.withColumnRenamed("prediction", "rf_pred")

    gbdt_pred = gbdt_model.transform(model_df.select("features", label_col))
    gbdt_pred = gbdt_pred.withColumnRenamed("prediction", "gbdt_pred")

    _id = model_df.select("id").withColumnRenamed("id", "Row").toPandas()
    final_pred = rf_pred.join(gbdt_pred, how="inner", on = "features") \
                        .withColumn("prediction", 
                                        func.round((func.col("rf_pred") + func.col("gbdt_pred")) / 2, 0)
                                        )
    pred_df = final_pred.select(label_col, "prediction").toPandas()

    pred_df = pd.concat([_id, pred_df], axis=1).reset_index(drop=True)
    pred_df = pred_df.sort_values(["Row"])

    pred_df.to_csv("./data/final_pred.csv")