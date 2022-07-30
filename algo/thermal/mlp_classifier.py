from algo.common.baseAlgo import BaseAlgo, BaseAlgoParams
from utils.datautils import load_for_mlp
from utils.datautils import scaling
from utils.mlputils import MLP_Classifier
# from utils.visutils import Confusion_Matrix
from sklearn.metrics import confusion_matrix
import logging
from sklearn.neural_network import MLPClassifier


class MLPClassificationParams(BaseAlgoParams):
    def __init__(self):
        super(MLPClassificationParams, self).__init__()
        self.train_csv_path = None
        self.test_csv_path = None
        self.hidden_layer_sizes = None
        self.activation = None
        self.max_iter = None

    def serialize(self):
        super(MLPClassificationParams, self).serialize()
        dict.__init__(self, MLPClassificationParams={"train_csv_path": self.train_csv_path,
                                                     "test_csv_path": self.test_csv_path,
                                                     "hidden_layer_sizes": self.hidden_layer_sizes,
                                                     "activation": self.activation, "max_iter": self.max_iter})

    def deserialize(self, data):
        super(MLPClassificationParams, self).deserialize(data)
        self.train_csv_path = data["AlgoParams"]["MLPClassificationParams"]["train_csv_path"]
        self.test_csv_path = data["AlgoParams"]["MLPClassificationParams"]["test_csv_path"]
        self.hidden_layer_sizes = data["AlgoParams"]["MLPClassificationParams"]["hidden_layer_sizes"]
        self.activation = data["AlgoParams"]["MLPClassificationParams"]["activation"]
        self.max_iter = int(data["AlgoParams"]["MLPClassificationParams"]["max_iter"])

        dict.__init__(self, train_csv_path=self.train_csv_path, test_csv_path=self.test_csv_path,
                      hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, max_iter=self.max_iter)


class MLPClassification(BaseAlgo):
    def __init__(self, AlgoParams: object = None) -> object:
        super(MLPClassification, self).__init__()
        self.name = "MLPClassification"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input=None):

        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting mlp_classifier")
            hidden_layer_sizes = tuple(int(num) for num in
                                       self.AlgoParams.hidden_layer_sizes.replace('(', '').replace(')', '').replace(
                                           '...', '').split(','))
            # Load data:
            (x_train, y_train, x_test, y_test) = load_for_mlp(self.AlgoParams.train_csv_path,
                                                              self.AlgoParams.test_csv_path,MLP=True)
            (x_trainscaled, x_testscaled) = scaling(x_train, x_test)

            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=self.AlgoParams.activation,
                                max_iter=self.AlgoParams.max_iter)

            # Results of mlp_classifier:
            [score, classification_report, y_pred] = MLP_Classifier(clf, x_trainscaled, y_train, x_testscaled, y_test)
            Confusion_Matrix = confusion_matrix(y_test, y_pred)

            ############################################################################################################################
            logging.info("Finished mlp_classifier")
        else:
            output = loaded

        # algo output dict
        mlp_classifier_output_dict = {'score': score, 'classification_report': classification_report,
                                      'Confusion_Matrix': Confusion_Matrix}

        if self.inputData == 'None':
            output = mlp_classifier_output_dict
            self.save_to_file(output)
        else:
            output: None = mlp_classifier_output_dict.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(MLPClassification, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, MLPClassification={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(MLPClassification, self).deserialize(data)
        self.name = data["MLPClassification"]["name"]
        self.AlgoParams = MLPClassificationParams()
        self.AlgoParams.deserialize(data)
        dict.__init__(self, name=self.name, MLPClassificationParams=self.AlgoParams)