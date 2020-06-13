#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:57:29 2020

@author: hardysmbp
"""
import os
import csv
import shutil
i=0
train=[]
with open('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train.csv', 'r') as file:
    reader = csv.reader(file, delimiter = ',')
    for row in reader:
        train.append(row)
        

for j in range(2500,2529):
    if train[j][1] == '0': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/normal')
        shutil.move(froml,tol) 
        
    if train[j][1] == '1': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/void')
        shutil.move(froml,tol) 
        
    if train[j][1] == '2': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/horizontal_defect')
        shutil.move(froml,tol) 
        
    if train[j][1] == '3': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/vertical_defect')
        shutil.move(froml,tol) 
        
    if train[j][1] == '4': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/edge_defect')
        shutil.move(froml,tol) 
        
    if train[j][1] == '5': 
        froml = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/train/',train[j][0])
        tol = os.path.join('/Users/hardysmbp/Documents/ML/Final/ML_final_project_NTHU/data/test_images/particle')
        shutil.move(froml,tol) 