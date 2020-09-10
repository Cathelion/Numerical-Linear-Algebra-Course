# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:24:31 2020

@author: duyp9
"""


import numpy as np 

class Orthogonalizationasd:
    
    def __init__(self,A):
        self.A=A
        
        
        
        def qr_factorization(A):
            m,n = A.shape 
            Q = np.zeros((m,n))
            R = np.zeros((m,n))
         
         
            for j in range(n):
                v = A[:,j]
             
                for i in range(j-1):
                    q = Q[:, i]
                    R[i,j] = q.dot(v)
                    v = v - R[i,j] * q
                    
                    norm = np.linalg.norm(v)
                    Q[:, j] = v / norm
                    R[j, j] = norm
            return(Q, R)
          
                    
        A = np.random.rand(13, 10) * 1000
        Q, R = qr_factorization(A)

        Q.shape, R.shape
        np.abs((A - Q.dot(R)).sum()) < 1e-6             