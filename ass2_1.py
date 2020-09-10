#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:31:32 2020

@author: florian
"""
import numpy as np


class Ortho:
    
    def __init__(self,A):
        self.A = A
        self.m = np.shape(A)[0]
        self.n = np.shape(A)[1]
        
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
    def project(self,u,v):
        return (v@u)/(u@u) * u

    def gramschmidt(self):
        A = self.A
        ortList = []
        for k in range(0,self.n):
            v_k = A[:,k]
            p_sum = np.zeros((self.m,))
            for i in range(k):
                p_sum += self.project(ortList[i],v_k)
            u_k = v_k - p_sum
            print("--",self.normalize(u_k))
            ortList.append(self.normalize(u_k))
        return np.array(ortList).T
            
A=  np.array([[1, 2,5],
       [3, 4,7],
       [5, 6,4]])

Ort1 = Ortho(A)
B = Ort1.gramschmidt()
print(A)
print(B)        
        
        
        
        
        
        
        
        
        
        

        