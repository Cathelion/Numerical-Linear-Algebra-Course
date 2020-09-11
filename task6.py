#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:54:54 2020

@author: flo
"""

import matplotlib.pyplot as plt
import scipy.linalg as sl
import numpy as np

A = plt.imread("kvinnagr.jpg",True)
m,n = np.shape(A)

U,s,V = sl.svd(A)
Sigma = sl.diagsvd(s,m,n)

B = (U @ Sigma) @ V

print("2-norm of Difference(original,SVD): ",np.linalg.norm(A-B,2))

plt.axis('off')
img = plt.imshow(B)
img.set_cmap('grey')