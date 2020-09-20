#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:40:20 2020

@author: florian
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))

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
    cos_x = np.sin(xdata)
    sin_2x = np.sin(2*xdata)
    cos_2x = np.sin(2*xdata)
    return np.column_stack((sin_x,cos_x,sin_2x, cos_2x))

x,y = process_data()
plt.plot(x,y, marker='x')
X = prepare(x)
coeff = sp.linalg.lstsq(X,y,rcond=None)

print(X @ coeff[0] - y)

x2 = np.linspace(0,12,1000)
y2 = 2 * np.sin(x2) + 5 * np.cos(x2) + 10 * np.sin(2*x2) + 5 * np.cos(2*x2)
plt.plot(x2,y2)









