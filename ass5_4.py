#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:13:17 2020

@author: florian
Numerical Linear Algebra Assignment 5 Task 4
Bisection Algorithm for eigenvalues
"""
import numpy as np
import scipy.linalg as sl

"""private functions"""
# finding the number of sign changes in sequence p_i(x)
# same as number of eig below x
def sign_change(A, x):
    n = A.shape[0]
    count = 0
    
    # starting values
    p_m1 = A[0,0] - x
    p_m2 = 1

    var_sign = p_m2   # keep track if above or below x-axis
    sign = np.sign(p_m1)
    
    # checking for the sign 
    # only count a sign change when really crossing the x-axis
    # so from + or 0 to - and from - or 0 to +
    # but not from + or - to 0
    if sign == 1 or sign ==-1:
        prod = sign * var_sign   # see if diff from prrevious
        if prod == -1:
            count+=1
            var_sign = var_sign * -1   # flip var_sign, xaxis crossed
    
    # loop over the sequence p_i(x)
    for k in range(2,n+1):
        p_k = (A[k-1,k-1] - x)*p_m1 - A[k-2,k-1]**2 * p_m2
        sign = np.sign(p_k)
        
        # sign change checking again
        if sign == 1 or sign ==-1:
            prod = sign * var_sign   # see if diff from prrevious
            if prod == -1:
                count+=1
                var_sign = var_sign * -1   # flip var_sign, xaxis crossed
        
        # update values
        p_m2 = p_m1 
        p_m1 = p_k
    return count, p_k

# find the biggest possible interval where the 
# eigenvalues could be located with gerschgorin disks
def findBiggestInterval(A):
    a = np.diag(A)
    b_1 = np.concatenate((np.array([0]),np.diag(A,1)))
    b_2 = np.concatenate((np.diag(A,1),np.array([0])))
    b = np.max(a+np.abs(b_1)+np.abs(b_2))    
    a = np.min(a-np.abs(b_1)-np.abs(b_2))
    return a,b

# bisection method
def bisection_lambda_i(A,i,a,b,tol=1e-8):
    if i > A.shape[0]:
        raise Exception("i > n, does not work")
        
    if np.abs(b-a) < tol:   # if interval is small enough, eig is found
        return (a+b)/2
    
    c = (a+b)/2
    num_zeros, p_n = sign_change(A, c)
    if p_n == 0:   # if c is an eig by accident return directly
        return c
    
    # otherwise perform bisection
    if num_zeros >= i:
        return bisection_lambda_i(A, i, a, c)
    else:
        return bisection_lambda_i(A, i, c, b)



"""public functions"""
# some nicer UX  
def findEigBisection(A,i,tol=1e-8):
    A = sl.hessenberg(A)
    a,b = findBiggestInterval(A)
    return bisection_lambda_i(A, i, a, b)    

# gives the number of eigenvalues in an interval
def findNumEig(A,a,b):
    A = sl.hessenberg(A)
    return sign_change(A, b)[0] - sign_change(A, a)[0]     
    

A = np.array([[1,1,0,0],[1,0,1,0],[0,1,2,1],[0,0,1,-1]])
A = np.random.rand(100,100)
A = A + A.T

print("All eigs: ",list(np.linalg.eig(A)[0]))
print("smallest Eig: ", findEigBisection(A, 1))
print("2nd smallest Eig: ", findEigBisection(A, 2))
print("biggest Eig: ", findEigBisection(A, 100))
print("negative Eigs: ",findNumEig(A, -1000, 0))