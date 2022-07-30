import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
def plt_2d_pca(X_pca, y, save_path,title):
    """
        :param X_pca: Pandas series of reduced dimenshionaly data
        :param Diagnosis: Pandas series of Diagnosis labels
    """
    fig = plt.figure(figsize=(8, 8))
    t = np.linspace(0, 13, num=14)
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y == 35.8, 0], X_pca[y == 35.8, 1], c=np.full((1, len(X_pca[y == 35.8, 0])), t[0]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.2')
    ax.scatter(X_pca[y == 36.0, 0], X_pca[y == 36.0, 1], c=np.full((1, len(X_pca[y == 36.0, 0])), t[0]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.2')
    ax.scatter(X_pca[y == 36.1, 0], X_pca[y == 36.1, 1], c=np.full((1, len(X_pca[y == 36.1, 0])), t[0]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.2')
    ax.scatter(X_pca[y == 36.2, 0], X_pca[y == 36.2, 1], c=np.full((1, len(X_pca[y == 36.2, 0])), t[0]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.2')
    ax.scatter(X_pca[y == 36.3, 0], X_pca[y == 36.3, 1], c=np.full((1, len(X_pca[y == 36.3, 0])), t[1]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.3')
    ax.scatter(X_pca[y == 36.4, 0], X_pca[y == 36.4, 1], c=np.full((1, len(X_pca[y == 36.4, 0])), t[2]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.4')
    ax.scatter(X_pca[y == 36.5, 0], X_pca[y == 36.5, 1], c=np.full((1, len(X_pca[y == 36.5, 0])), t[3]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.5')
    ax.scatter(X_pca[y == 36.6, 0], X_pca[y == 36.6, 1], c=np.full((1, len(X_pca[y == 36.6, 0])), t[4]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.6')
    ax.scatter(X_pca[y == 36.7, 0], X_pca[y == 36.7, 1], c=np.full((1, len(X_pca[y == 36.7, 0])), t[5]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.7')
    ax.scatter(X_pca[y == 36.8, 0], X_pca[y == 36.8, 1], c=np.full((1, len(X_pca[y == 36.8, 0])), t[6]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.8')
    ax.scatter(X_pca[y == 36.9, 0], X_pca[y == 36.9, 1], c=np.full((1, len(X_pca[y == 36.9, 0])), t[7]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='36.9')
    ax.scatter(X_pca[y == 37.0, 0], X_pca[y == 37.0, 1], c=np.full((1, len(X_pca[y == 37.0, 0])), t[8]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.0')
    ax.scatter(X_pca[y == 37.1, 0], X_pca[y == 37.1, 1], c=np.full((1, len(X_pca[y == 37.1, 0])), t[9]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.1')
    ax.scatter(X_pca[y == 37.2, 0], X_pca[y == 37.2, 1], c=np.full((1, len(X_pca[y == 37.2, 0])), t[10]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.2')
    ax.scatter(X_pca[y == 37.3, 0], X_pca[y == 37.3, 1], c=np.full((1, len(X_pca[y == 37.3, 0])), t[11]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.3')
    ax.scatter(X_pca[y == 37.4, 0], X_pca[y == 37.4, 1], c=np.full((1, len(X_pca[y == 37.4, 0])), t[12]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.4')
    ax.scatter(X_pca[y == 37.5, 0], X_pca[y == 37.5, 1], c=np.full((1, len(X_pca[y == 37.5, 0])), t[13]),
               norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm', label='37.5')
    ax.legend()
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    plt.savefig(os.path.join(save_path,title+".png"))
    plt.show()

    return

def plt_3d_orig(X_pca, y, feat_name,save_path,title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 13, num=14)
    ax.scatter(X_pca[y == 35.8, 0], X_pca[y == 35.8, 1], X_pca[y == 35.8, 2],
               c=np.full((1, len(X_pca[y == 35.8, 0])), t[0]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='35.8')
    ax.scatter(X_pca[y == 36.0, 0], X_pca[y == 36.0, 1], X_pca[y == 36.0, 2],
               c=np.full((1, len(X_pca[y == 36.0, 0])), t[0]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.0')
    ax.scatter(X_pca[y == 36.1, 0], X_pca[y == 36.1, 1], X_pca[y == 36.1, 2],
               c=np.full((1, len(X_pca[y == 36.1, 0])), t[0]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.1')
    ax.scatter(X_pca[y == 36.2, 0], X_pca[y == 36.2, 1], X_pca[y == 36.2, 2],
               c=np.full((1, len(X_pca[y == 36.2, 0])), t[0]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.2')
    ax.scatter(X_pca[y == 36.3, 0], X_pca[y == 36.3, 1], X_pca[y == 36.3, 2],
               c=np.full((1, len(X_pca[y == 36.3, 0])), t[1]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.3')
    ax.scatter(X_pca[y == 36.4, 0], X_pca[y == 36.4, 1], X_pca[y == 36.4, 2],
               c=np.full((1, len(X_pca[y == 36.4, 0])), t[2]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.4')
    ax.scatter(X_pca[y == 36.5, 0], X_pca[y == 36.5, 1], X_pca[y == 36.5, 2],
               c=np.full((1, len(X_pca[y == 36.5, 0])), t[3]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.5')
    ax.scatter(X_pca[y == 36.6, 0], X_pca[y == 36.6, 1], X_pca[y == 36.6, 2],
               c=np.full((1, len(X_pca[y == 36.6, 0])), t[4]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.6')
    ax.scatter(X_pca[y == 36.7, 0], X_pca[y == 36.7, 1], X_pca[y == 36.7, 2],
               c=np.full((1, len(X_pca[y == 36.7, 0])), t[5]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.7')
    ax.scatter(X_pca[y == 36.8, 0], X_pca[y == 36.8, 1], X_pca[y == 36.8, 2],
               c=np.full((1, len(X_pca[y == 36.8, 0])), t[6]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.8')
    ax.scatter(X_pca[y == 36.9, 0], X_pca[y == 36.9, 1], X_pca[y == 36.9, 2],
               c=np.full((1, len(X_pca[y == 36.9, 0])), t[7]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='36.9')
    ax.scatter(X_pca[y == 37.0, 0], X_pca[y == 37.0, 1], X_pca[y == 37.0, 2],
               c=np.full((1, len(X_pca[y == 37.0, 0])), t[8]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='37.0')
    ax.scatter(X_pca[y == 37.1, 0], X_pca[y == 37.1, 1], X_pca[y == 37.1, 2],
               c=np.full((1, len(X_pca[y == 37.1, 0])), t[9]), norm=plt.Normalize(vmin=0, vmax=13), cmap='coolwarm',
               label='37.1')
    # ax.scatter(X_pca[y == 37.2, 0], X_pca[y == 37.2, 1], X_pca[y == 37.2, 2], c=np.full((1, len(X_pca[y == 37.2, 0])), t[10]), norm = plt.Normalize(vmin=0, vmax=13),cmap='coolwarm', label='37.2')
    ax.scatter(X_pca[y == 37.3, 0], X_pca[y == 37.3, 1], X_pca[y == 37.3, 2],
               c=np.full((1, len(X_pca[y == 37.3, 0])), t[11]), norm=plt.Normalize(vmin=0, vmax=13),
               cmap='coolwarm', label='37.3')
    ax.scatter(X_pca[y == 37.4, 0], X_pca[y == 37.4, 1], X_pca[y == 37.4, 2],
               c=np.full((1, len(X_pca[y == 37.4, 0])), t[12]), norm=plt.Normalize(vmin=0, vmax=13),
               cmap='coolwarm', label='37.4')
    ax.scatter(X_pca[y == 37.5, 0], X_pca[y == 37.5, 1], X_pca[y == 37.5, 2],
               c=np.full((1, len(X_pca[y == 37.5, 0])), t[13]), norm=plt.Normalize(vmin=0, vmax=13),
               cmap='coolwarm', label='37.5')

    ax.set_xlabel(feat_name[0])
    ax.set_ylabel(feat_name[1])
    ax.set_zlabel(feat_name[2])
    ax.legend(loc="best")
    plt.suptitle(title)
    plt.savefig(os.path.join(save_path,title+".png"))
    plt.show()

    return


def tsne_preprocess(df ,n_comp ,full=True):
    # PARAM df: Data frame
    # PARAM full: boolean for checking on full data or just one frame for each patient
    # PARAM n_comp: number of components for tsne reduction
    if not full:
        # Visualizing just one frame of each patient
        df.rename({'Unnamed: 0': 'Name'}, axis='columns', inplace=True)
        df.drop_duplicates(subset="Name", keep='first', inplace=True)
    # save the labels into a variable label.
    label = df['Temp']
    # Drop the label feature and store the pixel data in d.
    d = df.drop("Temp", axis=1)
    # Data-preprocessing: Standardizing the data
    standardized_data = StandardScaler().fit_transform(d)
    model = TSNE(n_components=n_comp, random_state=0)
    tsne_data = model.fit_transform(standardized_data)
    tsne_data = np.vstack((tsne_data.T, label)).T
    if n_comp==2:
        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    elif n_comp==3:
        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2" ,"Dim_3", "label"))
    else:
        return exec('incompatible n_comp')

    return  tsne_df ,tsne_data
