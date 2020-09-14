#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:31:32 2020

@author: florian
"""
import numpy as np
import matplotlib.pyplot as plt

class Ortho:
    
    def __init__(self,A):
        self.A = A
        self.m = np.shape(A)[0]
        self.n = np.shape(A)[1]
    
    # help function     
    def normalize(self,v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    
    # projection
    def project(self,u,v):
        return (v@u)/(u@u) * u
    
    # orthogonalization with GS
    def gramschmidt(self):
        A = self.A
        ortList = []
        for k in range(0,self.n):
            v_k = A[:,k]
            p_sum = np.zeros((self.m,))
            for i in range(k):
                p_sum += self.project(ortList[i],v_k)
            u_k = v_k - p_sum
            ortList.append(self.normalize(u_k))
        return np.array(ortList).T
    
    # checks GS for accuracy
    def test(self):
        Q = self.gramschmidt()
        Z = Q.T @ Q
        norm1 = (np.linalg.norm(Q,2)-1)/self.n *self.m
        norm2 = (np.linalg.norm(np.eye(self.n) - Z, 2))/self.n *self.m
        eig1 = np.max(np.abs(np.linalg.eig(Z)[0]-np.ones(self.n))) 
        det = np.abs(np.abs(np.linalg.det(Z))-1)
        return norm1,norm2,eig1,det
    
    # help fcn to compute householder matrix
    def house(self,A):
        a=A[:,0]
        m = A.shape[0]
        ahat = np.array( [np.linalg.norm(a)] + ( m-1 )*[0.] )
        v=a- ahat
        v=v/ np.linalg.norm (v)
        Q=np.eye( m )- 2* np.outer (v , v )
        return Q
    
    # computes QR factorization
    def qr(self):
        A = self.A
        m = self.m
        n = self.n
        A_i = A
        Q = np.eye(m)
        for i in range(n):
            H_ii = self.house(A_i[i:,i:])
            H_i = np.eye(m)
            H_i[i:,i:] = H_ii
            A_i = H_i @ A_i
            Q = Q @ H_i.T
        return Q[:,:n], A_i[:n,:]
            
        
# test own QR against built-in
def QR_test():
    N = [1,10,100,500,1000,2000]
    for n in N:
        A = np.random.rand(n+2,n)
        Ort1 = Ortho(A)
        Q1, R1 = Ort1.qr()
        A_err = np.linalg.norm(A-Q1@R1,2)
        QI_err = np.linalg.norm(Q1.T@Q1-np.identity(n),2)
        Q2, R2 = np.linalg.qr(A)
        Q_err = np.linalg.norm(np.abs(Q1)-np.abs(Q2),2)
        R_err = np.linalg.norm(np.abs(R1)-np.abs(R2),2)
        print("Dimension:",n,"\n >A-QR error: ", A_err,"\n >I-QQ.T err: ",QI_err, "\n >Q error: ", Q_err, "\n >R error: ", R_err, "\n\n")
 
# testing and plotting of the errors of GS            
def test_and_plot():
    N = [1,10,100,500,1000,2000]
    data =  []
    for n in N:
        A = np.random.rand(n+2,n)
        Ort1 = Ortho(A)
        data.append(Ort1.test())
        print(n)


    for i in range(len(data)):
        data[i]=list(data[i])
    
    data= np.array(data)
    
    plt.figure(0)
    plt.plot(N, data[:,0],'x', label="2-norm")
    plt.xlabel("Dimension")
    plt.ylabel("Difference of 2-norm to 1")
    plt.legend()
    
    plt.figure(1)
    plt.plot(N, data[:,1],'x', label="Derivation from Identity")
    plt.xlabel("Dimension")
    plt.ylabel("Difference of to Identity")
    plt.legend()
    
    plt.figure(2)
    plt.plot(N, data[:,2],'x', label="Biggest mistake in eigenvalue")
    plt.xlabel("Dimension")
    plt.ylabel("Difference to eigenvalue 1")
    plt.legend()
            
    plt.figure(3)
    plt.plot(N, data[:,3],'x', label="Determinant")
    plt.xlabel("Dimension")
    plt.ylabel("Difference to determinant")
    plt.legend()
        
        
        
        
        
        
        

        