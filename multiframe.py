#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:46:27 2018

@author: cer
"""
from tkinter import Tk, ttk, Frame, Button, Label, Entry, Text, Checkbutton, Radiobutton,\
     Scale, Listbox, Menu, N, E, S, W, HORIZONTAL, END, FALSE, IntVar, StringVar
#from tkinter import BOTH, RIGHT, RAISED, messagebox as box, PhotoImage
#import os

import numpy as np
import matplotlib.pyplot as plt

import read_text_file_omega as rd
import read_header_line as rhl

import scatterg as a2dscatter
import scatterg3D as a3dscatter
import Normalize_Features as NF
import Randomize_training_samples as RTS
#import Shot_Attrib_NoiseStatistics as SA

class mainframe(Tk.frame):
    def __init__(self, parent, controller):
        Tk.Frame.__init__(self,parent)
        
        