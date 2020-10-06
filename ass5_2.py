#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 23:00:36 2020

@author: flo
"""

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl

"""does not work"""
# using householder reduction to hessenberg form
# if A hermitian becomes tridiagonal
def tridiag(A):   # A needs to be square
    m = A.shape[0]
    e_1 = np.zeros(m-1)
    e_1[0] = 1
    for k in range(m-2):
        x = A[k+1:,k]
        v_k = np.sign(x[0])* nl.norm(x) * e_1 + x
        v_k = v_k / nl.norm(v_k)
        A[k+1:,k:] = A[k+1:,k:] - 2*v_k*(v_k.T @ A[k+1:,k:])
        A[1:,k+1:] = A[1:,k+1:] - 2*(A[1:,k+1:] * v_k) @ v_k.T
    return A

def qr_shift(B):
    print('hello')
    m = B.shape[0]
    A = B.copy()
    
    if m==1:
        return A
    
    for i in range(50):
        u = A[m-1,m-1]
        Q,R = sl.qr(A-u*np.eye(m))
        A = R @ Q + u * np.eye(m)
        print("int;", A)
        for j in range(m-1):
            if np.abs(A[j,j+1]) < 10e-5:
                print('j:',j)
                A[j,j+1] = 0
                A[j+1,j] = 0
                
                A_1 = qr_shift(A[:j+1,:j+1])
                A_2 = qr_shift(A[j+1:,j+1:])
                return sl.block_diag(A_1,A_2)
    print("something wrong")
        

A = np.array([[1,2,0,0,0],[3,4,5,0,0],[0,6,7,8,0],[0,0,9,1,3],[0,0,0,6,8]])
print('input' , sl.hessenberg(A))
print(qr_shift(sl.hessenberg(A)))

