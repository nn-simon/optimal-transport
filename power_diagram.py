# %load /home/simon/omt-gu/omt-py/optimal-transport/power_diagram.py
import scipy as sci
from scipy.spatial import ConvexHull as convexhull
from scipy.spatial import Delaunay as delaunay
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

from Polygon import Polygon as polygon
import Polygon

def intersect_ray_polygon(ray, poly, scale, thresh = 0.001):
    a = ray[0:2]
    b = ray[0:2] + scale * ray[2:4]
    c = [b[0] + thresh, b[1] + thresh]
    tri = polygon(np.array([a, b, c]))
    region = poly & tri
    #print region.area()
    #print region
    npoints = region.nPoints()
    if npoints == 0:
        return []
    equal_thresh = 0.00001;
    for ii in range(npoints):
        if np.sum(np.abs(region[0][ii] - a) > equal_thresh):
            return region[0][ii]
    return []

def face_dual_uv(p):
    a = p[0, 1] * (p[1, 2] - p[2, 2]) + p[1, 1] * (p[2, 2] - p[0, 2]) + p[2, 1] * (p[0, 2] - p[1, 2])
    b = p[0, 2] * (p[1, 0] - p[2, 0]) + p[1, 2] * (p[2, 0] - p[0, 0]) + p[2, 2] * (p[0, 0] - p[1, 0])
    c = p[0, 0] * (p[1, 1] - p[2, 1]) + p[1, 0] * (p[2, 1] - p[0, 1]) + p[2, 0] * (p[0, 1] - p[1, 1])
    return np.array([-a/c/2, -b/c/2])

def plot_hull_2d(hull, ax):
    points = hull.points
    ax.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-')
    ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    ax.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    
    return ax

def plot_delaunay(faces, points, ax):
    ax.triplot(points[:,0], points[:,1], faces)
    for cnt, _face in enumerate(faces):
        heart = np.mean(points[_face], axis = 0)
        ax.text(heart[0], heart[1], str(cnt))

def _dict_update(dic, key, value):
        if key in dic:
            dic[key].append(value)
        else:
            dic.update({key: [value]})

def plot_pd_self(dpe, cells, ax):
    for key in cells:
        #print cells[key]
        ax.plot(dpe[cells[key], 0], dpe[cells[key], 1], 'r-')

_cmp = lambda x, y: [x, y] if x < y else [y, x]
ccw = lambda A, B, C: (B[0] - A[0]) * (C[1] - A[1]) > (B[1] - A[1]) * (C[0] - A[0]) #counterclockwise

def compute_edge_next(_face, point_inx, edge_cur):
    #print 'next:', _face, point_inx, edge_cur
    for ii in range(_face.size):
        if point_inx == _face[ii]:
            break
    #print ii
    edge1 = '%d_%d'%tuple(_cmp(_face[ii], _face[(ii + 1) % _face.size]))
    edge2 = '%d_%d'%tuple(_cmp(_face[ii], _face[ii - 1]))
    
    return edge1 if edge1 != edge_cur else edge2

def compute_cell(_face, pos, faces, edges_dic, cell):
    edge_start = '%d_%d'%tuple(_cmp(_face[pos], _face[(pos + 1) % _face.size]))
    edge_cur = edge_start
    flag_bd = False
    tri_last = cell[0]
    #print 'cell:', _face[pos]
    #print '\t',
    while True:
        #print edge_cur,
        tri_num = edges_dic[edge_cur]
        if len(tri_num) == 1:
            flag_bd = True
            break
        tri_cur = tri_num[0] if tri_num[0] != tri_last else tri_num[1]
        cell.append(tri_cur)
        edge_next = compute_edge_next(faces[tri_cur], _face[pos], edge_cur)
        if edge_next == edge_start:
            break
        edge_cur = edge_next
        tri_last = tri_cur

    #print '\n\tleft finished'
    if flag_bd == False:
        return flag_bd
    
    edge_cur= '%d_%d'%tuple(_cmp(_face[pos], _face[pos - 1]))
    tri_last = cell[0]
    while True:
        tri_num = edges_dic[edge_cur]
        if len(tri_num) == 1:
            break
        tri_cur = tri_num[0] if tri_num[0] != tri_last else tri_num[1]
        cell.insert(0, tri_cur)
        edge_next = compute_edge_next(faces[tri_cur], _face[pos], edge_cur)
        edge_cur = edge_next
        tri_last = tri_cur
        
    return flag_bd

def power_diagram_2d(points, h = None):
    npoints = points.shape[0]
    if h is None:
        h = np.zeros((points.shape[0], 1), np.float32)
    tp = np.sum(points * points, axis=1, keepdims=True)
    pl = np.concatenate((points, tp - h), axis = 1)
    hull = convexhull(pl)
    ind = hull.equations[:, 2] < 0
    nfaces = np.sum(ind)
    faces = hull.simplices[ind]
    dps = np.zeros((nfaces, 2), np.float32)
    dict_id_edge = dict()  # build a map dict in wich the key is edge and the value is the id of faces
    for cnt, _face in enumerate(faces):
        dps[cnt] = face_dual_uv(pl[_face])
        left, right = _cmp(_face[0], _face[1])
        _dict_update(dict_id_edge, '%d_%d'%(left, right), cnt)
        left, right = _cmp(_face[0], _face[2])
        _dict_update(dict_id_edge, '%d_%d'%(left, right), cnt)
        left, right = _cmp(_face[2], _face[1])
        _dict_update(dict_id_edge, '%d_%d'%(left, right), cnt)

    cells_dic = dict()
    flag_bd = np.zeros((npoints), np.int32)
    plot_delaunay(faces, points, ax2d)
    for cnt, _face in enumerate(faces):
        for pos in range(3):
            if _face[pos] in cells_dic:
                continue
            cell = [cnt]
            flag_bd[_face[pos]] = compute_cell(_face, pos, faces, dict_id_edge, cell)
            cells_dic.update({_face[pos]: cell})

    # construct a boundry
    dps_bd = np.zeros((np.sum(flag_bd), 2), np.float32)
    minxy = np.min(dps, axis = 0) - 1.0
    maxxy = np.max(dps, axis = 0) + 1.0
    scale = np.sqrt(np.sum((maxxy - minxy)**2))
    points_bd = np.array([[minxy[0], minxy[1]], [minxy[0], maxxy[1]], [maxxy[0], maxxy[1]], [maxxy[0], minxy[1]], [minxy[0], minxy[1]]])
    polygon_bd = polygon(points_bd)
    
    cnt = 0
    for ii in range(npoints):
        if flag_bd[ii]:
            flag_bd[ii] = nfaces + cnt
            cnt += 1

    bd_hull = convexhull(points)
    bd_edge = np.ones((npoints, 2), np.int32) * (-1)
    for edge in bd_hull.simplices:
        if bd_edge[edge[0]][0] == -1:
            bd_edge[edge[0]][0] = edge[1]
        else:
            bd_edge[edge[0]][1] = edge[1]
        if bd_edge[edge[1]][0] == -1:
            bd_edge[edge[1]][0] = edge[0]
        else:
            bd_edge[edge[1]][1] = edge[0]
    cnt = 0
    for ii in range(npoints):
        if flag_bd[ii] == 0 or bd_edge[ii][0] == -1:
            #print(flag_bd[ii], bd_edge[ii])
            continue
        cell = cells_dic[ii]
        if np.sum(bd_edge[ii][0] == faces[cell[0]]):
            B_inx = bd_edge[ii][0]
            C_inx = bd_edge[ii][1]
        else:
            B_inx = bd_edge[ii][1]
            C_inx = bd_edge[ii][0]
        if ccw(points[ii], points[B_inx], points[C_inx]):
            face_num = cell[0]
            other_inx = B_inx
            cell.insert(0, flag_bd[ii])
            cell.append(flag_bd[C_inx])
        else:
            face_num = cell[-1]
            other_inx = C_inx
            cell.append(flag_bd[ii])
            cell.insert(0, flag_bd[B_inx])
        vec = points[ii] - points[other_inx]
        vec /= np.sqrt(np.sum(vec**2))
        #print(ii, other_inx, B_inx, C_inx, face_num, vec)
        ray = np.array([dps[face_num, 0], dps[face_num, 1], -vec[1], vec[0]])
        point_inter = intersect_ray_polygon(ray, polygon_bd, scale)
        if point_inter == []:
            print('intersect_ray_polygon error!')
            exit()
        dps_bd[cnt] = np.array(point_inter)
        cnt += 1
    return dps, dps_bd, cells_dic, faces, points_bd

def verify(points, dps, cells_dic, h = None, thresh = 0.0001):
    dist2 = lambda x, y: np.sum((x - y)**2)
    dict_edge_cell = dict()
    npoints = points.shape[0]
    for ncell in range(npoints):
        if ncell in cells_dic:
            cell = cells_dic[ncell]
            for idx in range(len(cell) - 1):
                left, right = _cmp(cell[idx], cell[idx+1])
                _dict_update(dict_edge_cell, '%d_%d'%(left, right), ncell)
    if h is None:
        h = np.zeros(npoints, np.float32)
    for key in dict_edge_cell:
        if len(dict_edge_cell[key]) != 2:
            print(key, dict_edge_cell[key])
            continue
        id1, id2 = dict_edge_cell[key]
        spl = key.split('_')
        left = int(spl[0])
        right = int(spl[1])
        id1_l = dist2(points[id1], dps[left]) - h[id1]
        id1_r = dist2(points[id1], dps[right]) - h[id1]
        id2_l = dist2(points[id2], dps[left]) - h[id2]
        id2_r = dist2(points[id2], dps[right]) - h[id2]
        diff_l = id1_l - id2_l
        diff_r = id1_r - id2_r
        flag = 0
        if np.abs(diff_l) > thresh or np.abs(diff_r) > thresh:
            flag = 1
        print(flag, key, dict_edge_cell[key], diff_l, diff_r)

if __name__ == '__main__':
    points = np.random.rand(50, 2) * 2
    h = (np.random.rand(points.shape[0], 1) * 2 - 1) * 0.3
    dps, dps_bd, cells_dic, faces_pd, points_bd = power_diagram_2d(points, h)
    _dps = np.concatenate((dps, dps_bd), axis = 0)
    verify(points, _dps, cells_dic, h)
    scale = 4
    fig2d = plt.figure(figsize = (12 * scale, 11 * scale))
    ax2d = fig2d.add_subplot(111)
    plot_delaunay(faces_pd, points, ax2d)
    ax2d.plot(dps[:, 0], dps[:, 1], 'b*')
    for cnt in range(dps.shape[0]):
        ax2d.text(dps[cnt, 0], dps[cnt, 1], str(cnt), color = 'b')
    ax2d.plot(dps_bd[:, 0], dps_bd[:, 1], 'gx')
    plot_pd_self(_dps, cells_dic, ax2d)
    for cnt in range(dps_bd.shape[0]):
        ax2d.text(dps_bd[cnt, 0], dps_bd[cnt, 1], str(cnt + dps.shape[0]), color = 'c')
    ax2d.plot(points_bd[:, 0], points_bd[:, 1], 'c')
    plt.show()
