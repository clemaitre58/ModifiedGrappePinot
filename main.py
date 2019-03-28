import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
# from skcycling import Rider
# from skcycling.datasets import load_fit


def detec_aber(data, verbose=False):
    data = np.array(data)
    grad_data = np.diff(data)

    # creation list de zone de crossance
    l_aber = []
    for i in range(len(grad_data)):
        if grad_data[i] > 0:
            l_aber.append(i)

    if verbose:
        print('Nombre aberrations : ', len(l_aber))

    return l_aber, grad_data


def extract_great_data(data, index, l_aber):
    data = np.array(data)
    index = np.array(index)
    clean_data = []
    clean_index = []
    bmatch = False
    cpt_match = 0
    for i in range(len(index)):
        for j in range(len(l_aber)):
            if index[i] == l_aber[j]:
                # print('i : ', i, 'j', j, 'index[i] ', index[i], 'l_aber[j]',
                #      l_aber[j], 'match')
                bmatch = True
                cpt_match += 1
                break
        if bmatch is False:
            clean_index.append(index[i])
            clean_data.append(data[i])
        bmatch = False
    # print('cpt match', cpt_match)
    return clean_data, clean_index


if __name__ == '__main__':

    df = pd.read_csv('data_ppr.csv')
    ppr = df['ppr'].tolist()
    index = df.index.values
    l_aber, grad_data = detec_aber(ppr, verbose=True)
    c_ppr, c_index = extract_great_data(ppr, index, l_aber)

    c_index = np.array(c_index)
    c_index = c_index.reshape(-1, 1)
    c_ppr = np.array(c_ppr)

    # X -- index
    # Y -- ppr

    parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100]}
    svr = svm.SVR(kernel='rbf')
    clf = GridSearchCV(svr, parameters, cv=5)
    clf.fit(c_index, c_ppr)

    # plt.plot(np.linspace(0, 1, len(grad_data)), grad_data, 'r')
    # plt.show()
