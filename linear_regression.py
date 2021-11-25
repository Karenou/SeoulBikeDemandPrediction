from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from sklearn.model_selection import ParameterGrid
import math

def eval_lr(model, df, mode="Test"):
    """
    @param model: trained lr model
    @param df: df to be evaluated
    @param mode:
    """
    pred = model.evaluate(df)   
    rmse = math.sqrt(pred.meanSquaredError)
    r2 = pred.r2
    print("%s set, RMSE: %.2f" % (mode, rmse))
    print("%s set, R2: %.2f" % (mode, r2))
    print()
    return pred, rmse, r2


def linear_regression(train_df, val_df, test_df, label_col, params):
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
    best_rmse = 0
    grid = ParameterGrid(params)

    for param in grid:
        print(param)
        lin_Reg = LinearRegression(labelCol=label_col, **param)                
        lr_model = lin_Reg.fit(train_df)                             
        _, rmse, r2 = eval_lr(lr_model, val_df, "val")

        if  rmse < best_rmse:
            best_param = param
            best_rmse = rmse
            best_model = lr_model

    # print out result
    print("chosen hyperparameters", best_param)
    _, train_mse, train_r2 = eval_lr(best_model, train_df, "Train")
    _, test_mse, test_r2 = eval_lr(best_model, test_df, "Test")
    print("Linear regression model results:")
    print("Train set, RMSE: %.2f, R2 : %.2f" % (train_mse, train_r2))
    print("Test set, RMSE: %.2f, R2 : %.2f" % (test_mse, test_r2))

    return best_param, best_model