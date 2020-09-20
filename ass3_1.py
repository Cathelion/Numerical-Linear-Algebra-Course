#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:40:20 2020

@author: florian
"""
import scipy as sp
import scipy.linalg as sl
import numpy as np
import matplotlib.pyplot as plt
import time

def process_data():
    file1 = open('signal.dat', 'r') 
    lines = file1.readlines() 
    xVals = []
    yVals = []
    for l in lines:
        thing = l.split(',')
        thing[1] = thing[1][1:-2]
        xVals.append(float(thing[0]))
        yVals.append(float(thing[1]))
    return np.array(xVals), np.array(yVals)

def prepare(xdata):
    sin_x = np.sin(xdata)
    cos_x = np.cos(xdata)
    sin_2x = np.sin(2*xdata)
    cos_2x = np.cos(2*xdata)
    return np.column_stack((sin_x,cos_x,sin_2x, cos_2x))

def leastSquaresQR(X,b):
    Q,R = np.linalg.qr(X)
    rhs = Q.T @ b    # vector that is used to solve the LEQ
    n = len(rhs)
    xHat =  np.zeros(n)   #initialize
    for i in range(n):
        rSlice = R[n-1-i,n-i:]   #get the already calculated xvals
        xSlice = xHat[n-i:]   # get the corresponding x row
        xHat[n-1-i] = (rhs[n-1-i] - rSlice@xSlice) / R[n-1-i][n-1-i]
    return xHat

def leastSquaresSVD(X,b):
    m = X.shape[0]
    n = X.shape[1]
    U,s,V = np.linalg.svd(X)
    s_inv = np.zeros(n)
    for i in range(n):
        if s[i] != 0:
            s_inv[i] = 1/s[i]
    
    Sigma_inv = sl.diagsvd(s_inv,n,m)
    return V.T @ Sigma_inv @ U.T @ b
    


def plotSVD():
    plt.title('Model of Signal')
    plt.figure(figsize=(15,10))
    x,y = process_data()
    prepX = prepare(x)
    coeff2 = leastSquaresSVD(prepX, y)
    x1 = np.linspace(0,4*np.pi,1000)
    y1=coeff2[0] * np.sin(x1) + coeff2[1] * np.cos(x1) + coeff2[2] * np.sin(2*x1) + coeff2[3] * np.cos(2*x1)
    plt.scatter(x,y, marker='x')
    plt.plot(x1,y1, color='green')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.grid()
    print("SVD error: ", np.linalg.norm(prepX@coeff2 - y))
    
    
def plotQR():
    plt.title('Model of Signal')
    plt.figure(figsize=(15,10))
    x,y = process_data()
    prepX = prepare(x)
    coeff2 = leastSquaresQR(prepX, y)
    x1 = np.linspace(0,4*np.pi,1000)
    y1=coeff2[0] * np.sin(x1) + coeff2[1] * np.cos(x1) + coeff2[2] * np.sin(2*x1) + coeff2[3] * np.cos(2*x1)
    plt.scatter(x,y, marker='x')
    plt.plot(x1,y1, color='orange')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.grid()
    plt.grid()
    print("QR error: ", np.linalg.norm(prepX@coeff2 - y))


def plotBuiltin():
    plt.title('Model of Signal')
    plt.figure(figsize=(15,10))
    x,y = process_data()
    prepX = prepare(x)
    coeff = np.linalg.lstsq(prepX, y,rcond=None)[0]
    x1 = np.linspace(0,4*np.pi,1000)
    y1 = coeff[0] * np.sin(x1) + coeff[1] * np.cos(x1) + coeff[2] * np.sin(2*x1) + coeff[3] * np.cos(2*x1)
    plt.scatter(x,y, marker='x')
    plt.plot(x1,y1, color='red')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.grid()
    print("Builtin error: ", np.linalg.norm(prepX@coeff - y))

# some speedtests
def speedtest():
    x,y = process_data()
    prepX = prepare(x)

    start = time.time()
    for i in range(1000):
        coeff = np.linalg.lstsq(prepX, y,rcond=None)[0]
    end = time.time()
    print("\nSpeed for Builtin: ", (end-start)/1000)
    
    start = time.time()
    for i in range(1000):
        coeff = leastSquaresQR(prepX, y)
    end = time.time()
    print("Speed for QR: ", (end-start)/1000)
    
    start = time.time()
    for i in range(1000):
        coeff = leastSquaresSVD(prepX, y)
    end = time.time()
    print("Speed for SVD: ", (end-start)/1000)


plotSVD()
plotQR()
plotBuiltin()
speedtest()






