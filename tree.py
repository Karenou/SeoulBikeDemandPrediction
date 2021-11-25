from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from sklearn.model_selection import ParameterGrid
import pyspark.sql.functions as func


def evaluate_tree(model, df, mode="Test"):
    pred = model.transform(df)
    evaluator = RegressionEvaluator(labelCol="RentedBikeCount", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(pred, {evaluator.metricName: "r2"})
    print("%s set, RMSE: %.2f" % (mode, rmse))
    print("%s set, R2: %.2f" % (mode, r2))
    print()
    return pred, rmse, r2


def random_forest(train_df, val_df, test_df, label_col, params):
    """
    @param train_df: input train DataFrame
    @param val_df: input val DataFrame
    @param test_df: input test DataFrame
    @param label_col: str of target label name
    @param params: a dict of hyperparameters
    """
    # tune hyperparameter
    best_model = None
    best_param = None
    best_rmse = float('inf')
    grid = ParameterGrid(params)

    for param in grid:
        print(param)
        rf = RandomForestRegressor(featuresCol = "features", labelCol = label_col, predictionCol = "prediction", **param)                           
        model = rf.fit(train_df)
        _, rmse, r2 = evaluate_tree(model, val_df, mode="eval")

        if rmse < best_rmse:
            best_param = param
            best_rmse = rmse
            best_model = model

    # print out result
    print("chosen hyperparameters", best_param)
    _, train_mse, train_r2 = evaluate_tree(best_model, train_df, mode="Train")
    _, test_mse, test_r2 = evaluate_tree(best_model, test_df, mode="Test")
    print("Random forest model results:")
    print("Train set, RMSE: %.2f, R2 : %.2f" % (train_mse, train_r2))
    print("Test set, RMSE: %.2f, R2 : %.2f" % (test_mse, test_r2))
    return best_param, best_model


def gradient_boosting_tree(data, feature_col, label_col, params):
    """
    @param data: input spark DataFrame
    @param feature_col: list of feature columns
    @param label_col: str of target label name
    @param params: a dict of hyperparameters
    """
    # tune hyperparameter
    best_model = None
    best_param = None
    best_rmse = float('inf')
    grid = ParameterGrid(params)

    for param in grid:
        print(param)
        gbdt = GBTRegressor(featuresCol = "features", labelCol = label_col, predictionCol = "prediction", **param)                           
        model = gbdt.fit(train_df)
        _, rmse, r2 = evaluate_tree(model, val_df, mode="eval")

        if rmse < best_rmse:
            best_param = param
            best_rmse = rmse
            best_model = model

    # print out result
    print("chosen hyperparameters", best_param)
    _, train_mse, train_r2 = evaluate_tree(best_model, train_df, mode="Train")
    _, test_mse, test_r2 = evaluate_tree(best_model, test_df, mode="Test")
    print("GBDT model results:")
    print("Train set, RMSE: %.2f, R2 : %.2f" % (train_mse, train_r2))
    print("Test set, RMSE: %.2f, R2 : %.2f" % (test_mse, test_r2))

    return best_param, best_model
