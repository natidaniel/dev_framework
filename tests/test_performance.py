import sys
sys.path.append('../')

from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import csv
import plotly.express as px
import pandas as pd


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("gt_file", help="path to GT csv")
    arg_parse.add_argument("net_file", help="path to Net prediction output csv")
    arg_parse.add_argument("net_features_file", help="path to Net features")
    arg_parse.add_argument("test_name", help="name of the test (str)")
    arg_parse.add_argument("output_path", help="path to pipeline output path")
    args = arg_parse.parse_args()

    labels = []
    data = {}
    titles = []
    with open(args.gt_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for r in row:
                    titles.append(r)
                    data[r] = []
                line_count += 1
            else:
                for r, title in zip(row, titles):
                    data[title].append(r)

    labels = np.array(data[titles[1]]).astype(int)
    names = np.array(data[titles[2]])

    data = {}
    titles = []
    with open(args.net_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                for r in row:
                    titles.append(r)
                    data[r] = []
                line_count += 1
            else:
                for r, title in zip(row, titles):
                    if r == 'Invalid SMILES':
                        continue
                    data[title].append(r)

    preds = np.array(data[titles[1]]).astype(float)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    plt.plot(fpr,tpr)
    plt.show()

    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})

    opt_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds'].to_list()[0]
    label_preds = preds > opt_thresh

    c_m = confusion_matrix(labels, label_preds)
    print('false discovery = ', 100*c_m[0,1]/(c_m[0,1]+c_m[1,1]), '%')
    print('miss rate = ', 100*c_m[1,0]/(c_m[1,0]+c_m[1,1]), '%')
    print('AUC = ', roc_auc_score(labels, preds))

    # TSNE
    features = pickle.load(open(args.net_features_file, "rb"))
    X_embedded = TSNE(n_components=2).fit_transform(features)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()

    data.update({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'label':labels, 'names':names})
    df = pd.DataFrame.from_dict(data)
    fig = px.scatter(df, x='x', y='y', color='label', color_continuous_scale='Bluered_r',
                      title='TSNE of ' + args.test_name, hover_name='names')
    fig.show()
    fig.write_html(join(args.output_path + args.test_name + ".html"))