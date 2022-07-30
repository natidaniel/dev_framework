import torch
import pickle
import logging


class BaseObject(dict):
    def __init__(self):
        pass


class BaseAlgoParams(BaseObject):
    def __init__(self):
        super(BaseAlgoParams,self).__init__()
        self.input_file = "None"
        self.output_file = "None"

    def serialize(self):
        dict.__init__(self, BaseAlgoParams={"input_file": self.input_file, "output_file": self.output_file})

    def deserialize(self, data):
        self.input_file = data["AlgoParams"]["BaseAlgoParams"]["input_file"]
        self.output_file = data["AlgoParams"]["BaseAlgoParams"]["output_file"]
        dict.__init__(self, input_file = self.input_file, output_file = self.output_file)


class BaseAlgo(BaseObject):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super(BaseAlgo, self).__init__()
        self.inputData = None
        self.outputData = None
        self.useSavedData = False

    def run(self, input = None):
        logging.info("base is running ...")

    def save_to_file(self, output):
        string = self.AlgoParams.output_file
        if type(string) is str and string.endswith(".p"):
            with open(string, 'wb') as file:
               pickle.dump(output, file)
               logging.info('%s pickle is saved', self.name)

    def load_from_file(self):
        string = self.AlgoParams.input_file
        loaded = None
        if self.useSavedData is True and type(string) is str and string.endswith(".p"):
            with open(string, 'rb') as file:
               loaded = pickle.load(file)
               logging.info('%s pickle is loaded', self.name)
        return loaded

    def serialize(self):
        dict.__init__(self, BaseAlgo={"inputData": self.inputData, "outputData": self.outputData, "useSavedData": self.useSavedData})

    def deserialize(self, data):
        self.inputData = data["BaseAlgo"]["inputData"]
        self.outputData = data["BaseAlgo"]["outputData"]
        self.useSavedData = data["BaseAlgo"]["useSavedData"]
        dict.__init__(self, inputData = self.inputData, outputData = self.outputData, useSavedData = self.useSavedData)