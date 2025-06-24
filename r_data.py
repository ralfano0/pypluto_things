# Generate file with r coordinates (for later analysis)
# Author: Roberto Alfano

######################## imports ########################
import os
import sys
import pyPLUTO as pypl
import pyPLUTO.pload as pp
import pyPLUTO.Image as img
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

filenum = 0
################################################
dirpath = os.getcwd()+'/'
################################################

D = pp.pload(filenum, w_dir=dirpath, datatype='dbl')
x1r, x2r, x3r = np.meshgrid(D.x1, D.x2, D.x3)


x0 = 1.5
y0 = -3.0
z0 = 0.0

r = np.sqrt((x1r-x0)**2+(x2r-y0)**2+(x3r-z0)**2)
r = r.astype(np.float32)
r.tofile("r.bin")
