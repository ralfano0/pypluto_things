# Determine real mass and real energy of a
# PLUTO simulation 3D initial condition
# Author: Roberto Alfano
# Copyright (c) 2025, [TUO NOME]
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

import os
import sys
import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import *
import pyPLUTO as pypl
import pyPLUTO.pload as pp
import pyPLUTO.Image as img
import pandas as pd
import contextlib
from tqdm import tqdm
from numpy import nan


dirpath = os.getcwd()+'/'

nlinf = pypl.nlast_info(w_dir=dirpath,datatype='flt')
N = nlinf['nlast']
with contextlib.redirect_stdout(None):
  D = pp.pload(0, w_dir=dirpath,datatype='flt')

MU_SNR 	= 1.2889425/2. #!!

ti = 244.165
uv = 1.e8
ud = 1.67262171e-24
ul = 3.0856775807e18
kb = 1.3806505e-16
up = ud*(uv**2)
mu = MU_SNR

ti_str = str(ti) +'yrs'

plot(D.x1, ud*D.rho.T[:,int(D.n2/2),int(D.n3/2)])
yscale('log')
show()

print('density init max min = ',np.max(D.rho)*ud, np.min(D.rho)*ud," g/cm^3")
print('pressure init max min = ',np.max(D.prs)*up, np.min(D.prs)*up," dyn/cm^2")

print('density init max min = ',np.max(D.rho), np.min(D.rho)," CODE")
print('pressure init max min = ',np.max(D.prs), np.min(D.prs)," CODE")



V = (D.dx1[:])*(D.dx2[:])*(D.dx3[:])
#print(V)
En = D.tr2[:,:,:]*D.rho[:,:,:]*ud*(V*ul**3)*((D.vx1[:,:,:]*uv)**2+(D.vx2[:,:,:]*uv)**2+(D.vx3[:,:,:]*uv)**2)*0.5

realMass = D.tr2[:,:,:]*D.rho[:,:,:]*ud*(V*ul**3)/2.e33
windMass = D.tr3[:,:,:]*D.rho[:,:,:]*ud*(V*ul**3)/2.e33


print('Energy init() = ',En.sum(), " erg")
print('Mass init() = ',realMass.sum(), "M_sun")
print('wind Mass init() = ',windMass.sum(), "M_sun")


# Unefficient way
#En = np.zeros((D.n1,D.n2,D.n3))
#realMass = np.zeros((D.n1,D.n2,D.n3))
#
#
#for i in range(D.n1):
#	for j in range(D.n2):
#		for k in range(D.n3):
#			V = (D.dx1[i])*(D.dx2[j])*(D.dx3[k])
#			#print(V)
#			En[i,j,k] = D.tr2[i,j,k]*D.rho[i,j,k]*ud*(V*ul**3)*((D.vx1[i,j,k]*uv)**2+(D.vx2[i,j,k]*uv)**2+(D.vx3[i,j,k]*uv)**2)*0.5
#		
#			realMass[i,j,k] = D.tr2[i,j,k]*D.rho[i,j,k]*ud*(V*ul**3)/2.e33
#
#
#print('Energy init() = ',En.sum(), " erg")
#print('Mass init() = ',realMass.sum(), "M_sun")



