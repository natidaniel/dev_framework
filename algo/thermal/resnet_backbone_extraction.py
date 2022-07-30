from algo.common.baseAlgo import BaseAlgo, BaseAlgoParams
from os.path import join
from utils.resnetutils import Img2Vec
from utils.datautils import load_images_from_folder
from utils.visutils import tsne_2D_vis, tsne_3D_vis, pca_2D_vis, pca_3D_vis
import os
import pandas as pd
import logging


class ResNetExtractionParams(BaseAlgoParams):
    def __init__(self):
        super(ResNetExtractionParams, self).__init__()
        self.model_type = None
        self.inputImagePath = None
        self.patients_data_file = None
        self.img_manipulation_type = None
        self.outputPath = None
        self.save_embedding_file = False
        self.pca2D = False
        self.pca3D = False
        self.tsne2D = False
        self.tsne3D = False
        self.n_comp = None
        self.full = False

    def serialize(self):
        super(ResNetExtractionParams, self).serialize()
        dict.__init__(self, ResNetExtractionParams={"model_type":self.model_type, "inputImagePath":self.inputImagePath, "patients_data_file":self.patients_data_file, "img_manipulation_type":self.img_manipulation_type , "outputPath":self.outputPath, "save_embedding_file":self.save_embedding_file,
                                                     "pca2D":self.pca2D, "pca3D":self.pca3D, "tsne2D":self.tsne2D, "tsne3D":self.tsne3D, "n_comp":self.n_comp, "full":self.full})

    def deserialize(self, data):
        super(ResNetExtractionParams, self).deserialize(data)
        self.model_type = data["AlgoParams"]["ResNetExtractionParams"]["model_type"]
        self.inputImagePath = data["AlgoParams"]["ResNetExtractionParams"]["inputImagePath"]
        self.patients_data_file = data["AlgoParams"]["ResNetExtractionParams"]["patients_data_file"]
        self.img_manipulation_type = data["AlgoParams"]["ResNetExtractionParams"]["img_manipulation_type"]
        self.outputPath = data["AlgoParams"]["ResNetExtractionParams"]["outputPath"]
        self.save_embedding_file = data["AlgoParams"]["ResNetExtractionParams"]["save_embedding_file"]
        self.pca2D = data["AlgoParams"]["ResNetExtractionParams"]["pca2D"]
        self.pca3D = data["AlgoParams"]["ResNetExtractionParams"]["pca3D"]
        self.tsne2D = data["AlgoParams"]["ResNetExtractionParams"]["tsne2D"]
        self.tsne3D = data["AlgoParams"]["ResNetExtractionParams"]["tsne3D"]
        self.n_comp = data["AlgoParams"]["ResNetExtractionParams"]["n_comp"]
        self.full = data["AlgoParams"]["ResNetExtractionParams"]["full"]
        
        dict.__init__(self, model_type=self.model_type, inputImagePath=self.inputImagePath, patients_data_file=self.patients_data_file, img_manipulation_type=self.img_manipulation_type, outputPath=self.outputPath, save_embedding_file=self.save_embedding_file,
                       pca2D=self.pca2D, pca3D=self.pca3D, tsne2D=self.tsne2D, tsne3D=self.tsne3D, n_comp=self.n_comp, full=self.full)


class ResNetExtraction(BaseAlgo):
    def __init__(self, AlgoParams: object=None) -> object:
        super(ResNetExtraction, self).__init__()
        self.name = "ResNetExtraction"
        assert isinstance(AlgoParams, object)
        self.AlgoParams = AlgoParams

    def run(self, input = None):
        loaded = self.load_from_file()
        if loaded is None:
            logging.info("Starting extracting ResNet descriptors")

            # Initialize Img2Vec - choose model- 'alexnet'/'resnet18/34/50/101/152', cuda- True (gpu)/ False (cpu)
            img2vec = Img2Vec(model=self.AlgoParams.model_type)
            data = pd.read_excel(self.AlgoParams.patients_data_file, sheet_name='Patients data frame')
            data = data[['File Name', 'Patient Temperature']].set_index('File Name').transpose()

            # Initialize data frame
            df = pd.DataFrame()
            patients_imgs_paths = []

            subfolders = [f.path for f in os.scandir(self.AlgoParams.inputImagePath) if f.is_dir()]
            for folder in subfolders:
                folder_name = folder.split('/')[-1]
                logging.debug('patient ID: %s', folder_name)

                # Read list of images
                image_list, imgs_filenames = load_images_from_folder(os.path.join(folder, self.AlgoParams.img_manipulation_type))
                patients_imgs_paths.append(imgs_filenames)

                # Get list of vectors from img2vec, each vector represents one image returned as a torch FloatTensor
                temp_df = pd.DataFrame(img2vec.get_vec(image_list))
                temp_df.index = [folder_name] * len(image_list)
                #temp_df.index = imgs_filenames
                df = df.append(temp_df)

            t_list = df.index.to_list()
            for i, t in enumerate(t_list):
                t_list[i] = data[str(t)]['Patient Temperature']
            df.insert(0, 'Temp', t_list)

            # Save into output dir
            df.to_csv(join(self.AlgoParams.outputPath, self.AlgoParams.model_type + ".csv"))
            #PCA-2D visualization:
            if self.AlgoParams.pca2D == True:
                title = "PCA_2D_"+str(self.AlgoParams.model_type)
                pca_2D_vis(df, self.AlgoParams.outputPath, title)
            #PCA-3D visualization:
            if self.AlgoParams.pca3D == True:
                title = "PCA_3D_" + str(self.AlgoParams.model_type)
                pca_3D_vis(df, self.AlgoParams.outputPath, title)
            #TSNE-2D visualization:
            if self.AlgoParams.tsne2D == True:
                title = "TSNE_2D_" + str(self.AlgoParams.model_type)
                tsne_2D_vis(df, self.AlgoParams.n_comp, self.AlgoParams.full, title, self.AlgoParams.outputPath)
            #TSNE-3D visualization:
            if self.AlgoParams.tsne3D == True:
                title = "TSNE_3D_" + str(self.AlgoParams.model_type)
                tsne_3D_vis(df, self.AlgoParams.n_comp, self.AlgoParams.full, title, self.AlgoParams.outputPath)

            #####################################################################
            logging.info("Finished extracting ResNet descriptors")
        else:
            output = loaded

        # algo output dict
        resnet_algo_output_dict = {'patients_imgs_path': patients_imgs_paths, 'data_encodings': df}

        if self.inputData == 'None':
            output = resnet_algo_output_dict
            self.save_to_file(output)
        else:
            output: None = resnet_algo_output_dict.update(input)
            self.save_to_file(output)
        return output

    def serialize(self):
        super(ResNetExtraction, self).serialize()
        self.AlgoParams.serialize()
        dict.__init__(self, ResNetExtraction={"name": self.name}, AlgoParams=self.AlgoParams)

    def deserialize(self, data):
        super(ResNetExtraction, self).deserialize(data)
        self.name = data["ResNetExtraction"]["name"]
        self.AlgoParams = ResNetExtractionParams()
        self.AlgoParams.deserialize(data)
        dict.__init__(self, name=self.name, ResNetExtractionParams=self.AlgoParams)