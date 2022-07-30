from algo.common.baseAlgo import BaseAlgo, BaseAlgoParams
from utils.preprocessingutilis import path
from utils.preprocessingutilis import AVG
from utils.preprocessingutilis import black_box_temp
from utils.preprocessingutilis import thresh_circle_norm
from utils.preprocessingutilis import cascade
from utils.preprocessingutilis import CONTRAST
from utils.preprocessingutilis import JET
from utils.preprocessingutilis import ImgSpliting
from glob import glob
import os
import logging
import shutil


class ThermalPreprocessParams(BaseAlgoParams):
    def __init__(self):
        super(ThermalPreprocessParams, self).__init__()
        self.inputImagePath = None
        self.avg_rate = None
        self.outputPath = None
        self.notDetectedPath=None

    def serialize(self):
        super(ThermalPreprocessParams, self).serialize()
        dict.__init__(self, ThermalPreprocessParams={"inputImagePath":self.inputImagePath, "avg_rate" :self.avg_rate, "outputPath":self.outputPath,"notDetectedPath":self.notDetectedPath})

    def deserialize(self, data):
        super(ThermalPreprocessParams, self).deserialize(data)
        self.inputImagePath = data["AlgoParams"]["ThermalPreprocessParams"]["inputImagePath"]
        self.avg_rate = int(data["AlgoParams"]["ThermalPreprocessParams"]["avg_rate"])
        self.outputPath = data["AlgoParams"]["ThermalPreprocessParams"]["outputPath"]
        self.notDetectedPath=data["AlgoParams"]["ThermalPreprocessParams"]["notDetectedPath"]
        dict.__init__(self, inputImagePath=self.inputImagePath, avg_rate= self.avg_rate, outputPath=self.outputPath,notDetectedPath=self.notDetectedPath)



class ThermalPreprocess(BaseAlgo):
    def __init__(self, AlgoParams: object=None) -> object:
        super(ThermalPreprocess, self).__init__()
        self.name = "ThermalPreprocess"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input = None):
        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting preprocessing")
            not_detected_list=[]
            subfolders = [f.path for f in os.scandir(self.AlgoParams.inputImagePath) if f.is_dir()]
            for folder in subfolders:
                dict_path = path(folder,self.AlgoParams.avg_rate,self.AlgoParams.notDetectedPath)
                # Extracting patient black box temperature
                black_box = black_box_temp(folder)
                # Loading frames
                image_files = sorted(
                    glob('{}/*.tif'.format(dict_path['crop_path'])))  # images saved in tif format on crop folder
                # Loading Casscade model
                bboxes = cascade(image_files, dict_path)
                if len(bboxes) == 0:  # Face not detected, moving data to manual folder (in model will throw exception- try again)
                    shutil.move(folder, dict_path['notDetectedPath'])
                    logging.debug('Face not detected in ' + str(folder))
                    not_detected_list.append(folder)
                    continue
                # Performing threshold and circle segmentation + normlization to black box temperature
                thresh_circle_norm(folder, bboxes, image_files, black_box,dict_path)
                # Performing average
                AVG(folder, self.AlgoParams.avg_rate, dict_path)
                ###########################################################################################################################
            logging.info("Finished preprocessing")
        else:
            output = loaded

        # algo output dict
        preprocess_output_dict = {'Patients not detcted ': not_detected_list}

        if self.inputData == 'None':
            output = preprocess_output_dict
            self.save_to_file(output)
        else:
            output: None = preprocess_output_dict.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(ThermalPreprocess, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, ThermalPreprocess={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(ThermalPreprocess, self).deserialize(data)
        self.name = data["ThermalPreprocess"]["name"]
        self.AlgoParams = ThermalPreprocessParams()
        self.AlgoParams.deserialize(data)


class ThermalPreprocessAdvancedParams(BaseAlgoParams):
    def __init__(self):
        super(ThermalPreprocessAdvancedParams, self).__init__()
        self.inputImagePath = None
        self.outputPath = None
        self.avg_rate = None
        self.Contrast = None
        self.Colormap = None
        self.ColormapContrast = None
        self.Supface = None

    def serialize(self):
        super(ThermalPreprocessAdvancedParams, self).serialize()
        dict.__init__(self, ThermalPreprocessAdvancedParams={"inputImagePath":self.inputImagePath, "avg_rate":self.avg_rate,  "outputPath":self.outputPath,
                                                             "Contrast":self.Contrast, "Colormap":self.Colormap, "ColormapContrast":self.ColormapContrast,"Supface":self.Supface})

    def deserialize(self, data):
        super(ThermalPreprocessAdvancedParams, self).deserialize(data)

        self.inputImagePath = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["inputImagePath"]
        self.avg_rate = int(data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["avg_rate"])
        self.outputPath = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["outputPath"]
        self.Contrast = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["Contrast"]
        self.Colormap = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["Colormap"]
        self.ColormapContrast = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["ColormapContrast"]
        self.Supface = data["AlgoParams"]["ThermalPreprocessAdvancedParams"]["Supface"]

        dict.__init__(self, inputImagePath=self.inputImagePath,  avg_rate=self.avg_rate, outputPath=self.outputPath, Contrast=self.Contrast,
                      Colormap=self.Colormap, ColormapContrast=self.ColormapContrast, Supface=self.Supface)

class ThermalPreprocessAdvanced(BaseAlgo):
    def __init__(self, AlgoParams: object=None) -> object:
        super(ThermalPreprocessAdvanced, self).__init__()
        self.name = "ThermalPreprocessAdvanced"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input = None):
        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting preprocessing_advanced")

            subfolders = [f.path for f in os.scandir(self.AlgoParams.inputImagePath) if f.is_dir()]
            for folder in subfolders:
                paths_num = []
                paths_num.append(os.path.join(self.AlgoParams.inputImagePath+'\\'+ os.path.split(folder)[-1], 'contrast'))
                paths_num.append(os.path.join(self.AlgoParams.inputImagePath+'\\'+ os.path.split(folder)[-1], 'JET'))
                paths_num.append(os.path.join(self.AlgoParams.inputImagePath+'\\'+ os.path.split(folder)[-1], 'JET_contrast'))
                paths_num.append(os.path.join(self.AlgoParams.inputImagePath+'\\'+ os.path.split(folder)[-1], 'sup_face'))
                if not os.path.isdir(paths_num[0]):
                    os.mkdir(paths_num[0])
                if not os.path.isdir(paths_num[1]):
                    os.mkdir(paths_num[1])
                if not os.path.isdir(paths_num[2]):
                    os.mkdir(paths_num[2])
                if not os.path.isdir(paths_num[3]):
                    os.mkdir(paths_num[3])

                for subdir, dirs, files in os.walk(folder):
                    for dir in dirs:
                        if dir == 'Avg' + str(self.AlgoParams.avg_rate):
                            folder= os.path.join(subdir, 'Avg' + str(self.AlgoParams.avg_rate))
                            image_files_manipu = sorted(glob('{}/*.tif'.format(folder)))
                            Advanced_preprocess = []
                            ##contrast on normalized:
                            if self.AlgoParams.Contrast == True:
                                CONTRAST(image_files_manipu,paths_num[0])
                                Advanced_preprocess.append('Contrast')
                            ##colormap on normalized:
                            if self.AlgoParams.Colormap == True:
                                JET(image_files_manipu,paths_num[1])
                                Advanced_preprocess.append('Color_map')
                            ##colormap on contrasted:
                            if self.AlgoParams.ColormapContrast == True:
                                image_files_jetco = sorted(glob('{}/*.tif'.format(paths_num[0])))
                                JET(image_files_jetco,paths_num[2])
                                Advanced_preprocess.append('Colormap_Contrast')
                            ##Spliting images:
                            if self.AlgoParams.Supface == True:
                                ImgSpliting(image_files_manipu, paths_num[3])
                                Advanced_preprocess.append('Superior_face')
            ###########################################################################################################################
            logging.info("Finished preprocessing advanced")
        else:
            output = loaded

        # algo output dict
        preprocess_advanced_output_dict = {'Advanced Preprocess ': Advanced_preprocess}

        if self.inputData == 'None':
            output = preprocess_advanced_output_dict
            self.save_to_file(output)
        else:
            output: None = preprocess_advanced_output_dict.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(ThermalPreprocess, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, ThermalPreprocessAdvanced={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(ThermalPreprocessAdvanced, self).deserialize(data)
        self.name = data["ThermalPreprocessAdvanced"]["name"]
        self.AlgoParams = ThermalPreprocessAdvancedParams()
        self.AlgoParams.deserialize(data)
        dict.__init__(self, name=self.name, ThermalPreprocessAdvancedParams=self.AlgoParams)