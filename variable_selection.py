# encoding: utf-8
"""
@author: PangYuda、ChenJunjie、RuanChenzi
@contact: px7939592023@163.com
@time: 2023/12/01 19:37
@desc: 回归大作业
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
import argparse
import warnings
from copy import deepcopy
import logging
import random

warnings.filterwarnings("ignore")

pd.set_option("display.max_row", None)
# pd.set_option("display.max_column", None)


def args_parse():
    parse = argparse.ArgumentParser("波士顿房价--多元回归模型")
    parse.add_argument("--data_path", type=str, default="./HousingData.csv", help="原始数据文件路径")
    parse.add_argument("--y_transformation", type=str, default="log", help="对y进行转换的形式（函数）")
    parse.add_argument("--handle_nan", type=str, default="nearest_interpolate", help="插值方法，[drop, nearest_interpolate, mean_mode_interpolate]")
    parse.add_argument("--outlier_type", type=str, default="DEFITS", help="求异常值的方法，[DEFITS, abs_t, D]")
    parse.add_argument("--select_criteria", type=str, default="p_value", help="前向逐步回归中进行变量筛选的指标，[p_value, AIC]")
    args = parse.parse_args()
    return args


class Preprocess():
    def mean_mode_interpolate(self, data):
        """插值"""
        # mode_value = data[["ZN", "INDUS", "CHAS"]].mode()
        # mean_value = data[["CRIM", "AGE", "LSTAT"]]
        # data[["ZN", "INDUS", "CHAS"]] = data[["ZN", "INDUS", "CHAS"]].fillna(mode_value)
        # data[["CRIM", "AGE", "LSTAT"]] = data[["CRIM", "AGE", "LSTAT"]].fillna(mean_value)
        for col in ["ZN", "INDUS", "CHAS"]:
            data[col][data[col].isnull() == True] = data[col].mode()[0]
        # print(data[["ZN", "INDUS", "CHAS"]])
        for col in ["CRIM", "AGE", "LSTAT"]:
            data[col][data[col].isnull() == True] = data[col].mean()
        # for col in ["ZN"]:
            # data[col][data[col].isnull() == True] = np.nan
        data.dropna(inplace=True)
        return data

    def nearest_interpolate(self, data):
        # nearest interpolate
        # for col in data.columns:
        #     data[col][data[col].isnull() == True] = "9999"
        # data.replace("9999", np.nan, inplace=True)
        data = data.interpolate(method="nearest")
        data = data.fillna(data.mean())
        return data

    def drop_na(self, data, columns):
        for col in columns:
            data[col][data[col].isnull() == True] = np.nan
        data_dropna = data.dropna()
        return data_dropna

    def y_outlier(self, data, outlier_type):
        """异常值处理"""
        # DEFITS > 1: [365, 366, 368, 369, 372, 373, 375, 402, 406, 411, 413]
        if outlier_type == "DEFITS":
            outlier_index = [365, 366, 368, 369, 372, 373, 375, 402, 406, 411, 413]
            mid_list = []
            for outlier in outlier_index:
                if outlier in data.index:
                    mid_list.append(outlier)
            data = data.drop(mid_list, axis=0)
            data.index = range(len(data))
        # abs(t) > 2.5: [369, 372, 373, 401, 402, 413]
        if outlier_type == "abs_t":
            outlier_index = [369, 372, 373, 401, 402, 413]
            mid_list = []
            for outlier in outlier_index:
                if outlier in data.index:
                    mid_list.append(outlier)
            data = data.drop(mid_list, axis=0)
            data.index = range(len(data))
        # D > 0.05: [366, 368, 369, 402, 406, 411, 413]
        if outlier_type == "D":
            outlier_index = [366, 368, 369, 402, 406, 411, 413]
            mid_list = []
            for outlier in outlier_index:
                if outlier in data.index:
                    mid_list.append(outlier)
            data = data.drop(mid_list, axis=0)
            data.index = range(len(data))
        if outlier_type is None:
            return data
        return data

    def standardise(self, data, columns):
        s = StandardScaler()
        x = data[columns[:-1]]
        y = data[columns[-1]]
        x = s.fit_transform(x)
        logging.info(f"各变量样本均值：{[round(m, 4) for m in list(s.mean_)]}")
        logging.info(f"各变量样本方差：{[round(v, 4) for v in list(s.var_)]}")
        x = pd.DataFrame(x, columns=columns[:-1])
        x["MEDV"] = y
        return x

    def y_transformation(self, data):
        y = data[data.columns[-1]]
        y_log = np.log(list(y))
        data[data.columns[-1]] = y_log
        return data


class Feature_selection():
    def multicollinearity(self, data):
        """VIF检验多重共线性，初步拆选特征"""
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        x = np.matrix(data[data.columns[:-1]])
        VIF_list = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
        # print(VIF_list)
        multicollinearity_feature_list = {}
        for vif, col in zip(VIF_list, data.columns):
            if vif > 5:
                multicollinearity_feature_list[col] = vif
        return multicollinearity_feature_list

    def forward_stepwise_regression(self, data, select_criteria):
        # 按相关系数从大到小排序
        # ['LSTAT', 'RM', 'TAX', 'CRIM', 'NOX', 'PTRATIO', 'RAD', 'B', 'ZN', 'DIS', 'CHAS', 'MEDV']
        # corr = data.iloc[:, :-1].corrwith(data["MEDV"])
        # sort_value = corr.abs().sort_values(ascending=False).index
        # data_new = data[sort_value]
        # data_new["MEDV"] = data["MEDV"]

        # 第一步：对每个X进行一元回归,按t值或p值大小排序
        x_all = data[data.columns[:-1]]
        x_all = sm.add_constant(x_all)
        Y = data[data.columns[-1]]
        model_all = sm.OLS(Y, x_all).fit()
        # ["LSTAT", "CRIM", "PTRATIO", "DIS", "RM", "RAD", "NOX", "TAX", "B", "CHAS", "ZN"]
        t_statistic = list(model_all.pvalues.sort_values(ascending=True).index)
        const_index = t_statistic.index("const")
        t_statistic.remove("const")
        p_values = list(model_all.pvalues.sort_values(ascending=True).values)
        del p_values[const_index]
        logging.info("\n")
        logging.info("--------------------------------------------------------------------------------")
        logging.info(f"当前总变量**{t_statistic}**")
        logging.info("--------------------------开始前向逐步回归--------------------------------")
        X = data[t_statistic]

        # 逐步回归
        features = X.columns
        selected_feature = [features[0]]
        logging.info("--------------------------------------------------------------------------------")
        if select_criteria == "p_value":
            logging.info(f"加入变量**{features[0]}**({p_values[0]}<{0.05})， 当前已选择变量**{selected_feature}**")
        # 单变量AIC值
        x_one = X[features[0]]
        x_one = sm.add_constant(x_one)
        model_one = sm.OLS(Y, x_one).fit()
        best_aic = model_one.aic
        if select_criteria == "AIC":
            logging.info(f"加入变量**{features[0]}**(AIC={best_aic})， 当前已选择变量**{selected_feature}**")
        criteria0 = Criteria(data=data[[features[0]] + ["MEDV"]])
        criteria0.criteria_concatenate(data)

        if select_criteria == "p_value":
            # 第二步：计算剩下所有变量加入后的p值，加入<0.05的最小的p值
            a_to_enter = 0.05
            a_to_remove = 0.05    # a_to_enter <= a_to_remove
            features_later = list(features[1:])
            for i in range(len(features[1:])):
                pvalue_list_add = []
                col_list_add = []
                for feature in features_later:
                    features_add = selected_feature + [feature]
                    x_add = X[features_add]
                    x_add = sm.add_constant(x_add)
                    model_add = sm.OLS(Y, x_add).fit()
                    p_value_add = model_add.pvalues[feature]   # p值选变量
                    pvalue_list_add.append(p_value_add)
                    col_list_add.append(feature)
                min_pvalue_index_add = pvalue_list_add.index(min(pvalue_list_add))
                min_pvalue_add = pvalue_list_add[min_pvalue_index_add]   # 最小p值
                # logging.info(f"{pvalue_list_add}")
                add_col = col_list_add[min_pvalue_index_add]   # 最小p值对应变量名
                if min_pvalue_add < a_to_enter:
                    selected_feature.append(add_col)
                    # 加入该变量后，需要在features_later中删除该变量，以免重复选择相同变量
                    features_later.remove(add_col)
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info(f"确认加入变量**{add_col}**({min_pvalue_add}<{a_to_enter})，当前已有变量**{selected_feature}**")
                    criteria1 = Criteria(data=data[selected_feature + ["MEDV"]])
                    criteria1.criteria_concatenate(data)
                    if len(selected_feature) > 1:
                        # 第三步：删除除上述刚加入的X以外的所有不显著的X（逐个删除）
                        features_drop = deepcopy(selected_feature)

                        # 所有变量删除一个变量是否显著  --  p值筛选
                        x_drop = X[features_drop]
                        x_drop = sm.add_constant(x_drop)
                        model_drop = sm.OLS(Y, x_drop).fit()
                        p_values_drop = list(model_drop.pvalues.sort_values(ascending=True).values)
                        features_drop = list(model_drop.pvalues.sort_values(ascending=True).index)
                        new_feature_index = features_drop.index(add_col)
                        del p_values_drop[new_feature_index]
                        del features_drop[new_feature_index]
                        for p, fea_drop in zip(p_values_drop, features_drop):
                            if p > a_to_remove:
                                selected_feature.remove(fea_drop)
                                logging.info("--------------------------------------------------------------------------------")
                                logging.info(f"删除变量**{fea_drop}**({p}>{a_to_remove})，当前已有变量**{selected_feature}**")
                                criteria2 = Criteria(data=x_drop.drop("const", axis=1))
                                criteria2.criteria_concatenate(data)
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info("\n")
                else:
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info(f"剩余的所有变量均不显著，当前已选择变量**{selected_feature}**")
                    break
            logging.info("--------------------------------------------------------------------------------")
            logging.info(f"********************{selected_feature}********************")
            return selected_feature
        if select_criteria == "AIC":
            # 第二步：计算剩下所有变量加入后的p值，加入<0.05的最小的p值
            a_to_enter = 0.05
            a_to_remove = 0.05  # a_to_enter <= a_to_remove
            features_later = list(features[1:])
            for i in range(len(features[1:])):
                aic_list_add = []
                col_list_add = []
                for feature in features_later:
                    features_add = selected_feature + [feature]
                    x_add = X[features_add]
                    x_add = sm.add_constant(x_add)
                    model_add = sm.OLS(Y, x_add).fit()
                    p_value_add = model_add.pvalues[feature]  # p值选变量
                    aic_add = model_add.aic  # AIC值选变量
                    aic_list_add.append(aic_add)
                    col_list_add.append(feature)
                # logging.info(f"{pvalue_list_add}")
                min_aic_index_add = aic_list_add.index(min(aic_list_add))
                min_aic_add = aic_list_add[min_aic_index_add]
                add_col = col_list_add[min_aic_index_add]  # 最小AIC值对应变量名
                if min_aic_add < best_aic:
                    selected_feature.append(add_col)
                    # 加入该变量后，需要在features_later中删除该变量，以免重复选择相同变量
                    features_later.remove(add_col)
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info(
                        f"确认加入变量**{add_col}**({min_aic_add}<{best_aic})，当前已有变量**{selected_feature}**")
                    criteria1 = Criteria(data=data[selected_feature + ["MEDV"]])
                    criteria1.criteria_concatenate(data)
                    best_aic = min_aic_add  # 将当前AIC值变成最优的AIC值
                    if len(selected_feature) > 1:
                        # 第三步：删除除上述刚加入的X以外的所有不显著的X（逐个删除）
                        features_drop = deepcopy(selected_feature)

                        # AIC值筛选
                        feas = deepcopy(selected_feature)
                        features_drop.remove(add_col)
                        for fea_drop in features_drop:
                            features_compare = [fea for fea in feas if (fea_drop != fea)]
                            x_drop = X[features_compare]
                            x_drop = sm.add_constant(x_drop)
                            model_drop = sm.OLS(Y, x_drop).fit()
                            aic_drop = model_drop.aic
                            if aic_drop < best_aic:
                                selected_feature.remove(fea_drop)
                                logging.info(
                                    "--------------------------------------------------------------------------------")
                                logging.info(
                                    f"删除变量**{fea_drop}**({aic_drop}<{best_aic})，当前已有变量**{selected_feature}**")
                                criteria2 = Criteria(data=data[selected_feature + ["MEDV"]])
                                criteria2.criteria_concatenate(data)
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info("\n")
                else:
                    logging.info("--------------------------------------------------------------------------------")
                    logging.info(f"剩余的所有变量均不显著，当前已选择变量**{selected_feature}**")
                    break
            logging.info("--------------------------------------------------------------------------------")
            logging.info(f"********************{selected_feature}********************")
            return selected_feature


class Regression():
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def least_squares_method(self):
        """矩阵法"""
        x = self.data[self.columns[:-1]]
        x.insert(0, "cons", [1]*len(x))
        y = self.data[self.columns[-1]]
        x = np.array(x)
        y = np.array(y)
        tmp = np.linalg.inv(np.matmul(x.T, x))
        tmp = np.matmul(tmp, x.T)
        b = np.matmul(tmp, y)
        return b

    def MLE(self):
        x = self.data[self.columns[:-1]]
        y = self.data[self.columns[-1]]
        x = sm.add_constant(x)
        ols = sm.OLS(y, x)
        model = ols.fit()
        return model


class Metrics():
    def __init__(self, data, columns, y_test, y_pred):
        self.data = data
        self.columns = columns
        self.y_test = y_test
        self.y_pred = y_pred

    def r2(self):
        return round(r2_score(self.y_test, self.y_pred), 4)

    def mse(self):
        y_test = list(self.y_test)
        y_pred = list(self.y_pred)
        # if args.y_transformation == "log":
        #     y_test = [np.exp(y1) for y1 in y_test]
        #     y_pred = [np.exp(y2) for y2 in y_pred]
        return round(mean_squared_error(y_test, y_pred), 4)

    def mae(self):
        return round(mean_absolute_error(self.y_test, self.y_pred), 4)

    def cross_validation(self, n_splits):
        x = self.data[self.columns[:-1]]
        y = self.data[self.columns[-1]]
        linear = LinearRegression()
        scoring = {"r2": make_scorer(r2_score),
                   "mse": make_scorer(mean_squared_error),
                   "mae": make_scorer(mean_absolute_error)}
        # scores = cross_val_score(linear, x, y, cv=5, scoring=scoring, n_jobs=-1)
        kf = KFold(n_splits=n_splits, random_state=123, shuffle=True)
        scores = cross_validate(linear, x, y, cv=kf, scoring=scoring)
        return scores


class Criteria():
    def __init__(self, data):
        self.data = data
        self.y_true = self.data.iloc[:, -1]
        multi_linear = LinearRegression()
        multi_linear.fit(self.data.iloc[:, :-1], self.data.iloc[:, -1])
        self.y_pred = multi_linear.predict(self.data.iloc[:, :-1])

    def Rp_square(self):
        p = len(self.data.columns)
        SSEp = np.sum(np.subtract(self.y_true, self.y_pred)**2)
        y_true_mean = np.mean(self.y_true)
        SSTO = np.sum(np.subtract(self.y_true, y_true_mean)**2)
        return 1 - SSEp / SSTO

    def Rap_square(self):
        n = len(self.data)
        p = len(self.data.columns)
        SSEp = np.sum(np.subtract(self.y_true, self.y_pred)**2)
        y_true_mean = np.mean(self.y_true)
        SSTO = np.sum(np.subtract(self.y_true, y_true_mean)**2)
        return 1 - (n-1) / (n-p) * SSEp / SSTO

    def Cp(self, origin_data):
        n = len(self.data)
        p = len(self.data.columns)
        P = 14   # 总参数数
        SSEp = np.sum(np.subtract(self.y_true, self.y_pred)**2)
        # 计算保留所以X时的MSE
        multi_linear_all = LinearRegression()
        multi_linear_all.fit(origin_data[origin_data.columns[:-1]], origin_data[origin_data.columns[-1]])
        y_pred_all = multi_linear_all.predict(origin_data[origin_data.columns[:-1]])
        MSE_P = np.sum(np.subtract(self.y_true, y_pred_all)**2) / (n - P)
        return SSEp / MSE_P - (n-2*p)

    def AIC(self):
        n = len(self.data)
        p = len(self.data.columns)
        SSEp = np.sum(np.subtract(self.y_true, self.y_pred)**2)
        return n * np.log(SSEp) - n * np.log(n) + 2 * p

    def SBC(self):
        n = len(self.data)
        p = len(self.data.columns)
        SSEp = np.sum(np.subtract(self.y_true, self.y_pred)**2)
        return n * np.log(SSEp) - n * np.log(n) + np.log(n) * p

    def PRESS(self):
        n = len(self.data)
        multi_linear_removei = LinearRegression()
        sum_press = 0
        for i, y_true_i in zip(range(n), self.y_true):
            data_new = self.data.drop(i, axis=0)
            # multi_linear_removei.fit(data_new[data_new.columns[:-1]], data_new[data_new.columns[-1]])
            # y_pred_i = multi_linear_removei.predict(self.data.loc[i, self.data.columns[:-1]])   # 不能单行预测
            x = data_new[data_new.columns[:-1]]
            y = data_new[data_new.columns[-1]]
            x = sm.add_constant(x)
            ols = sm.OLS(y, x)
            model = ols.fit()
            params = model.params
            x_pred = list(self.data.loc[i, self.data.columns[:-1]])
            x_pred.insert(0, 1)
            y_pred_i = np.sum(np.multiply(x_pred, params))
            sum_press += (y_true_i - y_pred_i)**2
        return sum_press

    def criteria_concatenate(self, origin_data):
        Rp_square = self.Rp_square()
        Rap_square = self.Rap_square()
        Cp = self.Cp(origin_data=origin_data)
        AIC = self.AIC()
        SBC = self.SBC()
        PRESS = self.PRESS()
        print("-" * 50)
        print("Rp_square=", round(Rp_square, 4))
        print("Rap_square=", round(Rap_square, 4))
        print("Cp=", round(Cp, 4))
        print("AIC=", round(AIC, 4))
        print("SBC=", round(SBC, 4))
        print("PRESS=", round(PRESS, 4))
        print("-" * 50)
        logging.info(f"Rp_square={round(Rp_square, 4)}, Rap_square={round(Rap_square, 4)}, Cp={round(Cp, 4)}, AIC={round(AIC, 4)}, SBC={round(SBC, 4)}, PRESS={round(PRESS, 4)}")
        # print("\n")


def main():
    """***********************读取配置参数**********************"""
    args = args_parse()
    """*******************************************************"""


    """**********************输出日志************************"""
    logging.basicConfig(filename="./logging/multi_regression_analysis({}-{}-{})_cv_finish.log".format(args.handle_nan, args.outlier_type, args.select_criteria),
                        filemode="w", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    """********************************************************"""


    """***********************读取数据***************************"""
    data = pd.read_csv(args.data_path)
    # print(data)
    # print(data.info())
    # print(data.describe())
    # print(data.isnull())
    columns = data.columns
    logging.info("---------------原始数据变量---------------")
    logging.info(f"------{list(columns[:-1])}------")
    logging.info("\n")
    # print(columns)
    """*********************************************************"""


    """*************************数据预处理**************************"""
    logging.info("--------------------数据预处理--------------------")
    preprocess = Preprocess()
    logging.info("-------------------缺失值处理-------------------")
    """缺失值处理"""
    if args.handle_nan == "mean_mode_interpolate":
        data = preprocess.mean_mode_interpolate(data)
        data.index = range(len(data))   # 不加这个会报Y有nan的错
    if args.handle_nan == "nearest_interpolate":
        data = preprocess.nearest_interpolate(data)
    if args.handle_nan == "drop":
        data = preprocess.drop_na(data, columns)
        data.index = range(len(data))
    # data[["TAX", "AGE", "B"]] = data[["TAX", "AGE", "B"]] / 100
    """异常值处理"""
    data = preprocess.y_outlier(data, outlier_type=args.outlier_type)
    logging.info("---------------------标准化---------------------")
    data = preprocess.standardise(data, columns)
    print("------------------y_new=log(y)------------------")
    logging.info("------------------y_new=log(y)------------------")
    logging.info("\n")
    data = preprocess.y_transformation(data)
    origin_data = deepcopy(data)
    # data.to_csv("./after_preprocess2.csv", columns=data.columns, encoding="utf-8")
    # data = pd.read_csv("../linear_regression/after_preprocess2.csv", index_col=0, header=0)
    x_origin = origin_data[origin_data.columns[:-1]]
    Y = origin_data[origin_data.columns[-1]]

    """特征筛选前的模型评价结果"""
    linear = LinearRegression()
    linear.fit(x_origin, Y)
    Y_pred = linear.predict(x_origin)
    metric1 = Metrics(data, data.columns, y_test=Y, y_pred=Y_pred)
    print("r2=", metric1.r2())
    print("mse=", metric1.mse())
    print("mae=", metric1.mae())
    logging.info("筛选特征前的模型评价结果：")
    logging.info(f"r2={metric1.r2()}")
    logging.info(f"mse={metric1.mse()}")
    logging.info(f"mae={metric1.mae()}")
    n_split_list = [5, 10]
    for n_split in n_split_list:
        score = metric1.cross_validation(n_splits=n_split)
        # print("cross_validation:", score)
        logging.info(f"cross_validation-kflod={n_split}:")
        logging.info("r2_mean={}".format(np.mean(score["test_r2"])))
        logging.info("mse_mean={}".format(np.mean(score["test_mse"])))
        logging.info("mae_mean={}".format(np.mean(score["test_mae"])))
    """"""

    # y变换前后的残差分布图
    # x_origin = sm.add_constant(x_origin)
    # model = sm.OLS(Y, x_origin).fit()
    # plt.figure(figsize=(10, 6))
    # sns.distplot(model.resid, color="navy")
    # plt.title('log price model: residual')
    # plt.show()

    # 多元回归找不显著变量
    # x_origin = sm.add_constant(x_origin)
    # model = sm.OLS(Y, x_origin).fit()
    # print(model.summary())
    # print(model.pvalues)
    # p_values = list(model.pvalues.values)
    # features = list(model.pvalues.index)
    # large_pvalue = [fea for p, fea in zip(p_values, features) if p > 0.05]
    # if "const" in large_pvalue:
    #     large_pvalue.remove("const")
    # 一元回归找不显著变量
    p_values = []
    t_values = []
    for x_col in x_origin.columns:
        x = sm.add_constant(x_origin[x_col])
        model = sm.OLS(Y, x).fit()
        p_values.append(model.pvalues[x_col])
        t_values.append(model.tvalues[x_col])
    large_pvalue = [fea for p, fea in zip(p_values, x_origin.columns) if p > 0.05]
    logging.info("--------------------------------------------------------------------------------")
    logging.info("一元回归t检验的p值和t值")
    logging.info(f"p={p_values}")
    logging.info(f"t={t_values}")
    """************************************************************"""


    """************************六大特征选择标准***********************"""
    criteria1 = Criteria(data)
    print("----------原始数据特征选择的六大标准：----------")
    logging.info("-------------------------原始数据特征选择的六大标准：--------------------------")
    criteria1.criteria_concatenate(origin_data=origin_data)
    logging.info("--------------------------------------------------------------------------------")
    logging.info("\n")
    """************************************************************"""


    """********************初筛：删除p值>0.05的特征********************"""
    # large_pvalue = ["AGE", "INDUS"]
    logging.info("-------------------------删除一元回归中p值>0.05的变量-------------------------")
    for large_p in large_pvalue:
        data_drop_large_p = data.drop(large_p, axis=1)
        columns = data_drop_large_p.columns
        """*****删除特征后的六大特征值*****"""
        print("----------删除{}后的六大标准：----------".format(large_p))
        logging.info(f"-------------------------删除{large_p}后的六大标准：-------------------------")
        criteria4 = Criteria(data_drop_large_p)
        criteria4.criteria_concatenate(origin_data=origin_data)
        logging.info("--------------------------------------------------------------------------------")
        """***************************"""
    data_drop_large_pvalue = data.drop(large_pvalue, axis=1)
    """*****删除特征后的六大特征值*****"""
    print("----------删除{}后的六大标准：----------".format(large_pvalue))
    logging.info(f"-------------------------删除{large_pvalue}后的六大标准：-------------------------")
    criteria5 = Criteria(data_drop_large_pvalue)
    criteria5.criteria_concatenate(origin_data=origin_data)
    logging.info("--------------------------------------------------------------------------------")
    logging.info("\n")
    # 实验结果表明删除这两个特征效果更好
    data = data.drop(large_pvalue, axis=1)
    # 计算当前AIC值，为forward stepwise regression的最佳AIC值，用以比较、筛选变量
    Y = data[data.columns[-1]]
    now_best_X = data[data.columns[:-1]]
    now_best_X = sm.add_constant(now_best_X)
    now_best_model = sm.OLS(Y, now_best_X).fit()
    """************************************************************"""


    """***************************特征选择***************************"""
    feature_selection = Feature_selection()
    # 多重共线性
    multicollinearity_feature_list = feature_selection.multicollinearity(data)
    multicollinearity_feature_keys = list(multicollinearity_feature_list.keys())
    # {"RAD": 7.212, "TAX": 8.512}
    print("大于5的vif：{}".format(multicollinearity_feature_keys))
    # data = data.drop(multicollinearity_feature_keys, axis=1)
    # 对VIF值>5的X逐个删除，比较六大标准的值
    for key in multicollinearity_feature_keys:
        data_drop = data.drop(key, axis=1)
        columns = data_drop.columns
        """*****删除特征后的六大特征值*****"""
        print("----------删除{}后的六大标准：----------".format(key))
        logging.info(f"----------删除多重共线性变量{key}----------")
        criteria2 = Criteria(data_drop)
        criteria2.criteria_concatenate(origin_data=origin_data)
        """***************************"""
    data_drop_all_multicollinearity = data.drop(multicollinearity_feature_keys, axis=1)
    # data = data.drop(multicollinearity_feature_keys[0], axis=1)
    logging.info(f"----------删除多重共线性变量{multicollinearity_feature_keys}----------")
    """*****删除特征后的六大特征值*****"""
    print("----------删除{}后的六大标准：----------".format(multicollinearity_feature_keys))
    criteria3 = Criteria(data_drop_all_multicollinearity)
    criteria3.criteria_concatenate(origin_data=origin_data)
    logging.info("删除多重共线性变量后六个指标中两个R方变小，剩下四个变大了，说明效果更差，说明多重共线性变量对模型影响并不大，可保留")
    """***************************"""

    # forward stepwise regression  --  前向逐步回归
    select_criteria = args.select_criteria
    selected_feature = feature_selection.forward_stepwise_regression(data, select_criteria)
    data = data[selected_feature + ["MEDV"]]
    criteria_final = Criteria(data)
    criteria_final.criteria_concatenate(origin_data=origin_data)
    logging.info("--------------------------------------------------------------------------------")
    """**************************************************************"""


    """****************************参数拟合****************************"""
    # regression = Regression(data, columns)
    # param1 = regression.least_squares_method()
    # models = regression.MLE()
    # print(models.summary())
    # print("pvalue:", models.pvalues.round(4))
    # print(param1)
    # print(param2)

    # x_train, x_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]],
    #                                                     test_size=0.2, random_state=123)

    model_list = ["linear_regression", "ridge"]
    for model_type in model_list:
        if model_type == "linear_regression":
            model = LinearRegression()
            logging.info("普通多元线性回归模型的结果：")
        if model_type == "ridge":
            model = Ridge(alpha=0.5)
            logging.info("岭回归模型的结果：")
        x_train, y_train = data[data.columns[:-1]], data[data.columns[-1]]
        model.fit(x_train, y_train)
        coef = list(model.coef_)
        cons = model.intercept_
        print("变量：", list(x_train.columns))
        print("系数：", coef)
        print("截距：", cons)
        logging.info(f"变量：{list(x_train.columns)}")
        logging.info(f"系数：{coef}")
        logging.info(f"截距：{cons}")
        y_pred = model.predict(x_train)
        """**************************************************************"""


        """****************************模型评价****************************"""
        metric2 = Metrics(data, data.columns, y_test=y_train, y_pred=y_pred)
        print("r2=", metric2.r2())
        print("mse=", metric2.mse())
        print("mae=", metric2.mae())
        logging.info("筛选特征后的模型评价结果：")
        logging.info(f"r2={metric2.r2()}")
        logging.info(f"mse={metric2.mse()}")
        logging.info(f"mae={metric2.mae()}")
        for n_split in n_split_list:
            score = metric2.cross_validation(n_splits=n_split)
            # print("cross_validation:", score)
            logging.info(f"cross_validation-kflod={n_split}:")
            logging.info("r2_mean={}".format(np.mean(score["test_r2"])))
            logging.info("mse_mean={}".format(np.mean(score["test_mse"])))
            logging.info("mae_mean={}".format(np.mean(score["test_mae"])))
        logging.info("--------------------------------------------------------------------------------")
    """**************************************************************"""



if __name__ == "__main__":
    args = args_parse()

    data = pd.read_csv(args.data_path)

    main()







