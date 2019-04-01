import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
# from sklearn import svm
from scipy import stats
# from sklearn.model_selection import GridSearchCV
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


def create_feature(data):
    data = np.array(data)
    d1 = np.diff(data, prepend=data[0])
    d2 = np.diff(d1, prepend=d1[0])
    d3 = np.diff(d2, prepend=d2[0])
    features = np.column_stack((data, d1, d2, d3))

    return features


def pinot_grappe(data, index, verbose=False):

    data_oi = np.array(data[10*60:])
    index_oi = np.array(index[10*60:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(index_oi,
                                                                   data_oi)

    y_predict = np.zeros(len(data_oi))
    y_predict = intercept + index_oi * slope
    std_predict = np.std(data_oi - y_predict)

    if verbose:
        print('Slope : ', slope)
        print('Intercept : ', intercept)
        print('R2 : ', r_value**2)
        print('p_value : ', p_value)

    return slope, intercept, std_predict


def pinot_grappe_wko(data, index, verbose=False):

    data_oi = np.array(data[13:])
    index_oi = np.array(index[13:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(index_oi,
                                                                   data_oi)
    y_predict = np.zeros(len(data_oi))
    y_predict = intercept + index_oi * slope
    std_predict = np.std(data_oi - y_predict)

    if verbose:
        print('Slope : ', slope)
        print('Intercept : ', intercept)
        print('R2 : ', r_value**2)
        print('p_value : ', p_value)

    return slope, intercept, std_predict


def lin_regress_ransac(data, index, verbose=False):

    data_oi = np.array(data[10*60:])
    index_oi = np.array(index[10*60:])

    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100)
    ransac.fit(index_oi.reshape(-1, 1), data_oi)
    ppr_ransac = ransac.predict(index_oi.reshape(-1, 1))

    x1 = index_oi[0]
    y1 = ppr_ransac[0]

    x2 = index_oi[1]
    y2 = ppr_ransac[1]

    slope = (y2 - y1) / (x2 - x1)
    intercept = (x2 * y1 - x1 * y2) / (x2 - x1)

    if verbose:
        print('Slope : ', slope)
        print('Intercept : ', intercept)

    return ppr_ransac, index_oi, slope, intercept


def create_ppr_wko(ppr, lut):
    ppr_wko = []
    index_wko = []
    for t in lut:
        if t <= len(ppr):
            ppr_wko.append(ppr[t])
            index_wko.append(t)
        else:
            break

    return ppr_wko, index_wko


if __name__ == '__main__':

    lut_wko = [1, 5, 30, 60, 180, 210, 240, 270, 300, 330, 360, 390, 420, 600,
               1200, 1800, 2700, 3600, 7200, 10800, 14400]

    df = pd.read_csv('data_ppr.csv')
    ppr = df['ppr'].tolist()
    index = df.index.values
    index = np.array(index)
    l_aber, grad_data = detec_aber(ppr, verbose=True)
    c_ppr, c_index = extract_great_data(ppr, index, l_aber)
    index = np.log(index)

    c_index = np.array(c_index)
    # c_index = c_index.reshape(-1, 1)
    c_ppr = np.array(c_ppr)

    c_index = np.log(c_index)

    # ----------------------------
    # Create wko subset
    # -----------------------------

    ppr_wko, index_wko = create_ppr_wko(c_ppr, lut_wko)

    index_wko = np.array(index_wko)
    index_wko = np.log(index_wko)

    # features = create_feature(c_ppr)

    # # X -- index
    # # Y -- ppr

    # # parameters = {'C': [0.01, 0.1, 1, 10, 100, 1000],
    # #               'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100]}
    # clf = svm.SVR(kernel='rbf', gamma=0.0001, C=1000)
    # # clf = GridSearchCV(svr, parameters, n_jobs=-1, verbose=2, cv=5)
    # clf.fit(features, c_ppr)
    # l_aber = np.array(l_aber)
    # l_aber_pred = create_feature(l_aber)
    # rtn_predict = clf.predict(l_aber_pred)

    # # plt.plot(np.linspace(0, 1, len(grad_data)), grad_data, 'r')
    # # plt.show()
    sl_wko, inter_wko, std_wko = pinot_grappe_wko(ppr_wko, index_wko,
                                                  verbose=True)
    sl, inter, std = pinot_grappe(c_ppr, c_index, verbose=True)
    # Y, X, sl_r, inter_r = lin_regress_ransac(c_ppr, c_index, verbose=True)

    # original data without resampling
    plt.plot(c_index[180:], c_ppr[180:], 'o', label='original data')
    # orignal data with resampling
    plt.plot(index_wko[4:], ppr_wko[4:], 'g^', label='original data')
    # grappe pinot witouth resampling
    plt.plot(c_index[180:], inter + sl*c_index[180:], 'r', label='fitted line')
    plt.plot(c_index[180:], (inter + 2 * std) + sl*c_index[180:],
             '--r', label='fitted line')
    plt.plot(c_index[180:], (inter - 2 * std) + sl*c_index[180:],
             '--r', label='fitted line')
    # grappe pinot with resampling
    plt.plot(c_index[180:], inter_wko + sl_wko*c_index[180:],
             'y', label='fitted line')
    plt.plot(c_index[180:], (inter_wko + 2 * std_wko) + sl_wko*c_index[180:],
             '--y', label='fitted line')
    plt.plot(c_index[180:], (inter_wko - 2 * std_wko) + sl_wko*c_index[180:],
             '--y', label='fitted line')
    # plt.plot(c_index[180:], inter_r + sl_r*c_index[180:], 'Y',
    #          label='ransac-based fitted')
    plt.show()
