#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:18:00 2017

@author: Honglei
"""
import sys
import os

## use try and except, not ruining the whole process 
## when there is an error

def Cat(filename):
    try:    
        f = open(filename)
        text = f.read()
        print ('---', filename)
        print (text)
    except IOError:
        #even some function
        #HandleError()
        print ('IO error:', filename)
        

def main():
    args = sys.argv[1:]
    for arg in args:
        Cat(arg)

if __name__ == '__main__':
    main()
        

### you can even import this exception.py and reuse it
## import exception.py
## dir(exception)
## help(exception)
## help(exception.Cat)
## contents between """ and """ will be printed in help document





