import json
import collections

from algo.common.baseAlgo import BaseAlgo
from algo.thermal.image_retrieval_extraction import ImageRetrievalExtractionParams, ImageRetrievalExtraction
from algo.thermal.resnet_backbone_extraction import ResNetExtractionParams, ResNetExtraction
from algo.preprocessing.pre_processing import ThermalPreprocessParams, ThermalPreprocess
from algo.thermal.mlp_regressor import MLPRegressionParams, MLPRegression
from algo.thermal.mlp_classifier import MLPClassificationParams, MLPClassification


class Pipeline(collections.deque):
    def __init__(self) -> object:
        super(Pipeline,self).__init__()

    def append(self, x: object) -> object:
        if isinstance(x, BaseAlgo):
            super(Pipeline,self).append(x)
            return True

    def run(self,iodata=None):
        for algo in self:
            iodata = algo.run(iodata)
        return iodata

    def serialize(self, file_name):
        length = super(Pipeline, self).__len__()
        curVal = 0
        json_str = '{\n'
        algo: object
        for algo in self:
            algo.serialize()
            keys = algo.keys()
            algoName = None

            for key in keys:
                if key is not 'BaseAlgo' and key is not 'AlgoParams':
                    algoName = key
                    break
            if algoName is None:
                print('serializing a pipeline an algo has no algoName. Raising.')
                raise

            alg_str = "\"" + algoName + "\": "
            json_str += alg_str
            json_str += json.dumps(algo, indent=4)
            curVal += 1
            if curVal is not length:
                json_str += ',\n'
            else:
                json_str += '\n'

        json_str += '}'
        file = open(file_name, "w+")
        file.write(json_str)
        file.close()

    def deserialize(self, file_name: object) -> object:
        self.clear()

        with open(file_name,'r') as read_file:
            data = json.load(read_file)

        for algo in data.keys():
            alg_data = data[algo]
            klass = globals()[algo]
            algo_instanse = klass()
            algo_instanse.deserialize(alg_data)
            self.append(algo_instanse)