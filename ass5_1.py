#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:56:55 2020

@author: florian
"""

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa

A = np.array([[5,0,0,-1],[1,0,-1,1],[-1.5,1,-2,1],[-1,1,3,-3]])

def perturb(A,p):
    B = np.copy(A)
    diago = np.diag(B)
    B = B * p
    np.fill_diagonal(B,diago)
    return B

def formatEig(A):
    eigVal = sl.eig(A)[0]
    x = eigVal.real
    y = eigVal.imag
    return x,y

plt.figure(figsize=(15,8))
plt.title('Interesting Eigenvalues')
plt.ylim(-3.5, 3.5)
plt.xlim(-7, 7)

def plotGer(A):
    for i in range(A.shape[0]):
        d_i = A[i,i]
        r_i = np.sum(A[i,:])-d_i
        circle1= plt.Circle((d_i.real, d_i.imag), radius=r_i,color='green',fill=None)
        plt.gcf().gca().add_artist(circle1)

def plotEig():
    x,y = formatEig(A)
    plt.scatter(x,y,s=200, color='red', label = 'eig of A')
    plt.scatter([5,0,-2,-3],[0,0,0,0],s=300,color="purple", label='diagonal elements')
    
    p=np.linspace(0,1,11)
    for i in p:
        M = perturb(A, i)
        x,y  = formatEig(M)
        if i ==0:
            plt.scatter(x, y, s=10, color='blue' , label='eig of p(A)')
        else:
            plt.scatter(x, y, s=10, color='blue')
    

plotGer(A)
plotEig()
# circle1=plt.Circle((0,0),.2,color='r')
# plt.gcf().gca().add_artist(circle1)
plt.legend()