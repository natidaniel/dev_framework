from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from utils.vis_preprocess import plt_2d_pca, plt_3d_orig, tsne_preprocess
import matplotlib.pyplot as plt
import seaborn as sn
import os


def tsne_2D_vis(df, n_comp, full, title, save_path):


    tsne_df,tsne_data=tsne_preprocess(df,n_comp=n_comp,full=full)
    # Ploting the result of full tsne
    sn.FacetGrid(tsne_df, hue ="label", size = 6,palette='rocket_r').map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.suptitle(title)
    plt.savefig(os.path.join(save_path,title+".png"))
    plt.show()
    return

def tsne_3D_vis(df, n_comp, full, title, save_path):


    tsne_df,tsne_data=tsne_preprocess(df,n_comp=n_comp,full=full)
    plt_3d_orig(tsne_data,df['Temp'],['Dim_1','Dim_2','Dim_3'],save_path,title=title)

    return

def pca_2D_vis(df, save_path, title):
    # Dimension reduction on all data:
    row, col = df.shape
    vectors = [[0] * col] * row

    for i in range(len(vectors)):
        vectors[i] = df.iloc[i].to_list()
    vectors = np.asarray(vectors)
    vectors = np.delete(vectors, np.s_[0:2], axis=1)
    # encode Y = Temperature
    Y_encoder = df["Temp"]

    pca = PCA(n_components=2, whiten=True)
    X_pca = pca.fit_transform(vectors)

    plt_2d_pca(X_pca,Y_encoder, save_path,title)

    return

def pca_3D_vis(df,save_path,title):
    # Dimension reduction on all data:
    row, col = df.shape
    vectors = [[0] * col] * row

    for i in range(len(vectors)):
        vectors[i] = df.iloc[i].to_list()
    vectors = np.asarray(vectors)
    vectors = np.delete(vectors, np.s_[0:2], axis=1)
    # encode Y = Temperature
    Y_encoder = df["Temp"]

    pca = PCA(n_components=3, whiten=True)
    X_pca = pca.fit_transform(vectors)

    feat_name = ['U1', 'U2', 'U3']
    plt_3d_orig(X_pca, Y_encoder, feat_name,save_path,title)

    return