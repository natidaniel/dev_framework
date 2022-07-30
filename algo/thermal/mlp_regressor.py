from algo.common.baseAlgo import BaseAlgo, BaseAlgoParams
from utils.datautils import load_for_mlp
from utils.datautils import scaling
from utils.mlputils import MLP_Regressor
from glob import glob
import os
import logging
import shutil


class MLPRegressionParams(BaseAlgoParams):
    def __init__(self):
        super(MLPRegressionParams, self).__init__()
        self.train_csv_path = None
        self.test_csv_path = None
        self.hidden_layer_sizes = None
        self.activation = None
        self.solver = None
        self.learning_rate = None
        self.learning_rate_init = None
        self.power_t = None
        self.alpha = None
        self.max_iter = None
        self.early_stopping = None
        self.warm_start = None
        self.cv = None

    def serialize(self):
        super(MLPRegressionParams, self).serialize()
        dict.__init__(self,
                      MLPRegressionParams={"train_csv_path": self.train_csv_path, "test_csv_path": self.test_csv_path,
                                           "hidden_layer_sizes": self.hidden_layer_sizes, "activation": self.activation,
                                           "solver": self.solver, "learning_rate": self.learning_rate,
                                           "learning_rate_init": self.learning_rate_init, "power_t": self.power_t,
                                           "alpha": self.alpha, "max_iter": self.max_iter,
                                           "early_stopping": self.early_stopping, "warm_start": self.warm_start,
                                           "cv": self.cv})

    def deserialize(self, data):
        super(MLPRegressionParams, self).deserialize(data)
        self.train_csv_path = data["AlgoParams"]["MLPRegressionParams"]["train_csv_path"]
        self.test_csv_path = data["AlgoParams"]["MLPRegressionParams"]["test_csv_path"]
        self.hidden_layer_sizes = data["AlgoParams"]["MLPRegressionParams"]["hidden_layer_sizes"]
        self.activation = data["AlgoParams"]["MLPRegressionParams"]["activation"]
        self.solver = data["AlgoParams"]["MLPRegressionParams"]["solver"]
        self.learning_rate = data["AlgoParams"]["MLPRegressionParams"]["learning_rate"]
        self.learning_rate_init = data["AlgoParams"]["MLPRegressionParams"]["learning_rate_init"]
        self.power_t = data["AlgoParams"]["MLPRegressionParams"]["power_t"]
        self.alpha = data["AlgoParams"]["MLPRegressionParams"]["alpha"]
        self.max_iter = data["AlgoParams"]["MLPRegressionParams"]["max_iter"]
        self.early_stopping = data["AlgoParams"]["MLPRegressionParams"]["early_stopping"]
        self.warm_start = data["AlgoParams"]["MLPRegressionParams"]["warm_start"]
        self.cv = data["AlgoParams"]["MLPRegressionParams"]["cv"]

        dict.__init__(self, train_csv_path=self.train_csv_path, test_csv_path=self.test_csv_path,
                      hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver,
                      learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init,
                      power_t=self.power_t, alpha=self.alpha, max_iter=self.max_iter,
                      early_stopping=self.early_stopping, warm_start=self.warm_start, cv=self.cv)


class MLPRegression(BaseAlgo):
    def __init__(self, AlgoParams: object = None) -> object:
        super(MLPRegression, self).__init__()
        self.name = "MLPRegression"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input=None):

        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting mlp_regressor")
            hidden_layer_sizes = tuple(int(num) for num in
                                       self.AlgoParams.hidden_layer_sizes.replace('(', '').replace(')', '').replace(
                                           '...', '').split(','))
            # Load data:
            (x_train, y_train, x_test, y_test) = load_for_mlp(self.AlgoParams.train_csv_path,
                                                              self.AlgoParams.test_csv_path)
            (x_trainscaled, x_testscaled) = scaling(x_train, x_test)

            param_grid = {'hidden_layer_sizes': hidden_layer_sizes,
                          'activation': [self.AlgoParams.activation],
                          'solver': [self.AlgoParams.solver],
                          'learning_rate': [self.AlgoParams.learning_rate],
                          'learning_rate_init': [self.AlgoParams.learning_rate_init],
                          'power_t': [self.AlgoParams.power_t],
                          'alpha': [self.AlgoParams.alpha],
                          'max_iter': [self.AlgoParams.max_iter],
                          'early_stopping': [self.AlgoParams.early_stopping],
                          'warm_start': [self.AlgoParams.warm_start]}

            # Results of mlp_regressor:
            [r2_score, cv_results, best_params] = MLP_Regressor(param_grid, self.AlgoParams.cv, x_trainscaled, y_train,
                                                                x_testscaled, y_test)

            ############################################################################################################################
            logging.info("Finished mlp_regressor")
        else:
            output = loaded

        # algo output dict
        mlp_regressor_output_dict = {'r2_score_0.1': r2_score, 'cv_results': cv_results, 'best_params': best_params}

        if self.inputData == 'None':
            output = mlp_regressor_output_dict
            self.save_to_file(output)
        else:
            output: None = mlp_regressor_output_dict.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(MLPRegression, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, MLPRegression={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(MLPRegression, self).deserialize(data)
        self.name = data["MLPRegression"]["name"]
        self.AlgoParams = MLPRegressionParams()
        self.AlgoParams.deserialize(data)
        dict.__init__(self, name=self.name, MLPRegressionParams=self.AlgoParams)