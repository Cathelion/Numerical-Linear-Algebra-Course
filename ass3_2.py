# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:29:32 2020

@author: duyp9
"""
import numpy as np


class Ortho:
    
    def _init_(self,A):
        self.A  = A
        self.n = np.shape(A)[0]
        self.m = np.shape(A)[1]
        
    def norm(self,v):
        norm = np.linalg.norm(v)
        if norm == 0 :
            return v
        return v/norm
    
    
    def householder(self,A):
        Q = np.identity(self.m)
        R = np.copy(A)
        
        for i in range(self.n-1):
            v = np.copy(R[:,1])
            q = 0
            while q>i:
                v[q] = 0
                q  = q + 1
            v = np.transpose(np.array(v))
            H = np.identity(self.m) - 2*v*np.transpose(v)/Ortho.norm(v)**2
            R = np.dot(H,R) 
            Q = np.dot(H,Q) 
            
