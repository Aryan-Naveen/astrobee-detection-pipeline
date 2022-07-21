#!/usr/bin/env python3

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class Undistorter:
    def __init__(self, dims, K, dist):
        K_undist = K.copy()
        K_undist[0:2,2] = dims/2.
        #get set of x-y coordinates
        coords = np.mgrid[0:dims[0], 0:dims[1]].reshape((2, np.prod(dims)))
        #need to compute source coordinates on distorted image for undist image
        normalized_coords = np.linalg.solve(K_undist, np.vstack((coords, np.ones(coords.shape[1]))))

        #FOV model
        coeff1 = 1./dist
        coeff2 = 2 * np.tan(dist/2)
        rus = np.linalg.norm(normalized_coords[:2], axis=0)
        rds = np.arctan(rus * coeff2) * coeff1

        conv = np.ones(rus.shape)
        valid_pts = rus > 1e-5
        conv[valid_pts] = rds[valid_pts]/rus[valid_pts]

        scaled_coords = normalized_coords
        scaled_coords[0:2] *= conv
        dist_coords = np.matmul(K, scaled_coords)[:2, :].reshape((2, dims[0], dims[1]))
        self.dist_coords_ = np.transpose(dist_coords, axes=(0,2,1)).astype(np.float32)

    def undistort(self, img):
        undist_img = cv2.remap(img, self.dist_coords_[0], self.dist_coords_[1], cv2.INTER_CUBIC)
        return undist_img

if __name__=='__main__':
    #Bumble camera model
    dims = np.array([1280, 960])
    K = np.array([
      [608.8073, 0.0, 632.53684],
      [0.0, 607.61439, 549.08386],
      [0.0, 0.0, 1.0]])
    dist = 0.998693
    undist = Undistorter(dims, K, dist)

    img = cv2.imread('/media/ian/HDD1/ISSData/localization_bag1/imgs/1615399358.5696492.jpg')
    undist_img = undist.undistort(img)
    cv2.imshow('orig', img)
    cv2.imshow('undist', undist_img)
    cv2.waitKey(0)
