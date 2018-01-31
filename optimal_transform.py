# %load /home/simon/omt-gu/omt-py/optimal-transport/optimal_transform.py
import scipy as sci
import numpy as np
from scipy.spatial import ConvexHull as convexhull
from scipy.spatial import Delaunay as delaunay
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

from Polygon import Polygon as polygon
import Polygon

from power_diagram import power_diagram_2d, plot_delaunay, plot_pd_self

def plot_diagram(points, faces_pd, cells_dic, dps, dps_bd, points_bd, scale = 1.0, verbose = 0):
    _dps = np.concatenate((dps, dps_bd), axis = 0)
    fig2d = plt.figure(figsize = (12 * scale, 11 * scale))
    ax2d = fig2d.add_subplot(111)
    plot_delaunay(faces_pd, points, ax2d)
    ax2d.plot(dps[:, 0], dps[:, 1], 'b*')
    ax2d.plot(dps_bd[:, 0], dps_bd[:, 1], 'gx')
    plot_pd_self(_dps, cells_dic, ax2d)
    ax2d.plot(points_bd[:, 0], points_bd[:, 1], 'c')
    if verbose:
        for cnt in range(dps.shape[0]):
            ax2d.text(dps[cnt, 0], dps[cnt, 1], str(cnt), color = 'b')
        for cnt in range(dps_bd.shape[0]):
            ax2d.text(dps_bd[cnt, 0], dps_bd[cnt, 1], str(cnt + dps.shape[0]), color = 'c')
    plt.show()

def discrete_optimal_transform(points, delta, polygon_bd, scale, h = None, learnrate = 0.1, max_iter = 100, epsilon = 1e-5):
    '''
    Discrete Optimal Transport
    '''
    if h is None:
        h = np.zeros((points.shape[0], 1), np.float32)
    dps, dps_bd, cells_dic, faces_pd, new_h = power_diagram_2d(points, polygon_bd, scale, h, c = learnrate)
    _dps = np.concatenate((dps, dps_bd), axis = 0)
    area = np.sum(delta)
    for _iter in range(max_iter):
        G = calculate_gradient(points, _dps, cells_dic, polygon_bd)
        G = G / np.sum(G) * area
        D = G - delta
        if np.max(np.abs(D)) < epsilon:
            break
        dps, dps_bd, cells_dic, faces_pd, new_h = power_diagram_2d(points, polygon_bd, scale, new_h, D, c = learnrate)
        #print(_iter, new_h.T)
        print(G.T)
        _dps = np.concatenate((dps, dps_bd), axis = 0)
        if _iter % 20 == 0:
            plot_diagram(points, faces_pd, cells_dic, dps, dps_bd, points_bd, scale = 1.0)
    return new_h

def calculate_gradient(points, dps, cells_dic, polygon_bd):
    npoints = points.shape[0]
    D = np.zeros((npoints,1), np.float32)

    for npt in range(npoints):
        if npt not in cells_dic:
            continue
        cell = cells_dic[npt]
        poly = polygon(dps[cell])
        region = poly & polygon_bd
        D[npt] = region.area()
    return D

if __name__ == '__main__':
    npoints = 50
    points = np.random.random((npoints, 2))
    minxy = np.min(points, axis = 0) - 0.1
    maxxy = np.max(points, axis = 0) + 0.1
    scale = np.sqrt(np.sum((maxxy - minxy)**2))
    points_bd = np.array([[minxy[0], minxy[1]], [minxy[0], maxxy[1]], [maxxy[0], maxxy[1]], [maxxy[0], minxy[1]], [minxy[0], minxy[1]]])
    polygon_bd = polygon(points_bd)
    delta = (maxxy[0] - minxy[0]) * (maxxy[1] - minxy[1]) / npoints * np.ones((npoints, 1), np.float32)
    
    new_h = None
    new_h = discrete_optimal_transform(points, delta, polygon_bd, scale, new_h, learnrate = 0.05, max_iter = 2000, epsilon = 1e-5)

