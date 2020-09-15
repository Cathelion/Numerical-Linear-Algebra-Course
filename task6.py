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
B = U @ Sigma @ V

print("2-norm of Difference(original,SVD): ",np.linalg.norm(A-B,2))

plt.figure(figsize=(20,15))
plt.subplots_adjust() 
plt.subplot(2,3,1)
plt.axis('off')
plt.imshow(B/255.0, cmap='gray')
plt.title("original picture", fontsize=25)

def compress(r, A):
    U,s,V = sl.svd(A)
    Sigma = sl.diagsvd(s,m,n)
    U_hat = U[:,0:r]
    Sigma_hat = Sigma[0:r,0:r]
    V_hat = V[0:r,:]
    print(np.shape(U_hat@ Sigma_hat@ V_hat))
    return U_hat@ Sigma_hat@ V_hat


R = [5,10,25,50,100]
for r in R:
    plt.subplot(2,3,R.index(r)+2)
    plt.axis('off')
    plt.imshow(compress(r,A)/255 , cmap ='gray')
    plt.title("rank r="+str(r)+" approximation", fontsize=25)    
    







