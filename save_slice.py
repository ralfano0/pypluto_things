# Create slices data in binary files from
# PLUTO simulations
# Author: Roberto Alfano
# Copyright (c) 2025, [TUO NOME]
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

######################## imports ########################
import os
import sys
import matplotlib as mpl
from matplotlib import colors
from matplotlib import pyplot as plt
import pyPLUTO as pypl
import pyPLUTO.pload as pp
import pyPLUTO.Image as img
import pandas as pd
import contextlib
import numpy as np

######################## constants, fixed variables, unit measures ########################

mu = 1.2889425		#mean molecular weight			
ti = 0				#initial time
ud = 1.67262171e-24		#unit density (g)
ul = 3.0856775807e18		#unit length (cm)
uv = 1.e8			#unit velocity (cm/s)
up = (uv**2)*ud			#unit pressure (dyn/cm^2)
umag = uv*np.sqrt(4*np.pi*ud)	#unit magnetic field (G)
utyrs = (ul/uv)*3.17098e-8	#unit time (yrs)
kb = 1.3806505e-16		#Boltzmann constant in cgs
mp = 1.67262171e-24		#Mass of the proton in cgs

#filenum = 28
#filenum2 = 29
################################################

plt.rcParams.update({
#		"text.usetex": True,
#		"font.family": "sans-serif",
#		"font.sans-serif": "Helvetica",
	"font.size": 10
})
plt.rc('font',weight='bold')
#plt.rcParams['text.latex.preamble']=r'\usepackage{sfmath} \boldmath'

dirpath = '/scratch/ralfano/VER10_time/15M1F/'

props = dict(boxstyle='round',facecolor=None, alpha=0.0, edgecolor='white')	# properties of textbox

################################################
nlast = 56

for i in range(nlast+1):
	D = pp.pload(i, w_dir=dirpath, datatype='dbl')
	rho_img = D.rho[:,:,250].T
	temp_img = D.prs[:,:,250].T*up*mu*ud/(D.rho[:,:,250].T*ud*kb)

	rho_img.tofile("rho_img"+str(i)+".npy")
	temp_img.tofile("temp_img"+str(i)+".npy")
	tr2_img = D.tr2[:,:,250].T

	tr2_img.tofile("tr2_img"+str(i)+".npy")

	del D, tr2_img, rho_img, temp_img
