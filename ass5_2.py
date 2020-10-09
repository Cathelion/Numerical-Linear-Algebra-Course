#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:00:36 2020
@author: flo
"""

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import time
import matplotlib.pyplot as plt

"""does not work"""
# using householder reduction to hessenberg form
# if A hermitian becomes tridiagonal
def tridiag(A):   # A needs to be square
    m = A.shape[0]
    for k in range(m-2):
        x = A[k+1:,k]
        e_1 = np.zeros(x.shape[0])
        e_1[0] = 1
        v_k = np.sign(x[0])* nl.norm(x) * e_1 + x
        v_k = v_k / nl.norm(v_k)
        A[k+1:,k:] = A[k+1:,k:] - 2*np.outer(v_k , (v_k @ A[k+1:,k:]))
        A[:,k+1:] = A[:,k+1:] - 2*np.outer((A[:,k+1:] @ v_k), v_k)
    return A

def qr_shift(A):
    m = A.shape[0]
    
    # a 1x1 matrix is of course diagonal so return directly
    if m==1:
        return A,0
    
    # otherwise loop
    for i in range(50):
        # check if any off-diagonal entries are zero (within tol)
        # do this before qr computations
        for j in range(m-1):
            if np.abs(A[j,j+1])/np.abs(A[j,j]) < 1e-8:           
                # when only 2x2 matrix return directly as already orthogonal
                if (m==2):
                    A[j,j+1] = 0
                    A[j+1,j] = 0
                    return A, i   
                else:  # otherwise chop it up and do partwise
                    A_1,k = qr_shift(A[:j+1,:j+1])
                    A_2,j = qr_shift(A[j+1:,j+1:])
                    return sl.block_diag(A_1,A_2), i+k+j # stack blocks in the end
        
        # if no zeros on the off-diag are found 
        # do the shifting algorithm with the QR
        u = A[m-1,m-1]
        Q,R = sl.qr(A-u*np.eye(m))
        A = R @ Q + u * np.eye(m)
        
    
    # if no zeros show up after 50 loops, abort
    print("ERROR: the QR shifting does not produce zeros")

def getEig(A):
    M = qr_shift(sl.hessenberg(A))
    return np.diag(M)
    
counts = []
for n in range(1,100):
    dim =n 
    A = np.random.rand(dim,dim)
    A = A+A.T
    eig,count = qr_shift(tridiag(A))
    counts.append(count)

plt.figure(figsize=(10,8))
plt.xlabel('dimension')
plt.ylabel('iterations')
plt.title('Number of iterations')
plt.scatter(list(range(1,100)), counts, marker='x')

    
