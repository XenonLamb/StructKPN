import numpy as np
import numba as nb
from math import floor, atan2, pi, isnan, sqrt
import cv2



def sr_train_process(in_img, gt_img, in_GX, in_GY, ratio, h_hsize, f_hsize, w, Qangle, Qstrength, Qcoherence, stre, cohe, Q, V, mark, step=1):
    H, W = in_img.shape
    p_hsize = max(h_hsize, f_hsize)

    for i1 in range(p_hsize, H - p_hsize, step):
        for j1 in range(p_hsize, W - p_hsize, step):
            hash_idx1 = (slice(i1 - h_hsize, i1 + h_hsize + 1), slice(j1 - h_hsize, j1 + h_hsize + 1))
            filter_idx1 = (slice(i1 - f_hsize, i1 + f_hsize + 1), slice(j1 - f_hsize, j1 + f_hsize + 1))
            patch = in_img[filter_idx1]
            patchX = in_GX[hash_idx1]
            patchY = in_GY[hash_idx1]
            theta, lamda, u = hash_table(patchX, patchY, w, Qangle, Qstrength, Qcoherence, stre, cohe)
            # print(theta, lamda ,u)
            patch1 = patch.ravel()
            patchL = patch1.reshape((1, patch1.size))
            j = theta * Qstrength * Qcoherence + lamda * Qcoherence + u
            jx = np.int(j)
            A = np.dot(patchL.T, patchL)
            Q[jx] += A
            for r1 in range(ratio):
                for r2 in range(ratio):
                    pos = r1 * ratio + r2
                    b1 = patchL.T * gt_img[i1*ratio+r1, j1*ratio+r2]
                    b = b1.reshape((b1.size))
                    V[pos, jx] += b
            mark[jx] = mark[jx] + 1



def grad_patch(patch_x, patch_y):
    gx = patch_x.ravel()
    gy = patch_y.ravel()
    G = np.vstack((gx, gy)).T
    x = np.dot(G.T, G)
    w, v = np.linalg.eig(x)
    index = w.argsort()[::-1]
    w = w[index]
    v = v[:, index]
    theta = atan2(v[1, 0], v[0, 0])
    if theta < 0:
        theta = theta + pi
    theta = theta/pi
    lamda = sqrt(w[0])
    u = (np.sqrt(w[0]) - np.sqrt(w[1])) / (np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001)
    return theta, lamda, u


def main(img_path):
    imm =
    gx, gy =
