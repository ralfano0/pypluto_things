# Make plots and animated plots from PLUTO simulations
#Author: Roberto Alfano
# Copyright (c) 2025, [TUO NOME]
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/

import warnings

######################## ignore warnings ########################
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
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
	from tqdm import tqdm
	import numpy as np
	import argparse

	######################## plot dark background ########################

	plt.style.use("dark_background")
	plt.rcParams.update({
#		"text.usetex": True,
#		"font.family": "sans-serif",
#		"font.sans-serif": "Helvetica",
		"font.size": 18
	})

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
	
	######################## command line parsing ########################
	parser = argparse.ArgumentParser(description="plot cuts of 3D cartesian coordinates map of density [-r/--rho], temperature [-t/--tmp], pressure [-p/--prs] and velocities along x axis [-vx/--vx1], y axis [-vy/--vx2] or z axis [-vz,vx3]. It is possible to plot profile cuts (i.e. 1D cuts) by specifying the option [-pr/--profile] and, with the option [-m/--movie] or [-om/--onlymovie], one can respectively make the plots and the movie or just the movie on already existing plots. It is possible to make and show a single plot [-s/--single] or to make the plots for all the files in the folder [-a/--all]")

	#algorithm variables
	parser.add_argument("-s","--single",action='store_true', help='single file input, requires -n to be specified')
	parser.add_argument("-st","--selstep",help='specify remapping step in single mode')
	parser.add_argument("-pr","--profile",action='store_true', help='enable profile cut plot')
	parser.add_argument("-n","--nfile",help='specify file number to plot in single mode')
	parser.add_argument("-i","--ideal",help='plot ideal profiles in profile plots',action='store_true')
	parser.add_argument("-m","--movie",help='make movies',action='store_true')
	parser.add_argument("-om","--onlymovie",help='makes only the movie from existing plots',action='store_true')

	#physical quantities
	parser.add_argument("-r","--rho",help='plot density',action='store_true')
	parser.add_argument("-p","--prs",help='plot pressure',action='store_true')
	parser.add_argument("-t","--tmp",help='plot temperature',action='store_true')
	parser.add_argument("-vx","--vx1",help='plot x axis velocity',action='store_true')
	parser.add_argument("-vy","--vx2",help='plot y axis velocity',action='store_true')
	parser.add_argument("-vz","--vx3",help='plot z axis velocity',action='store_true')
	parser.add_argument("-t1","--tr1",help='plot tracer 1 (shock time)',action='store_true')
	parser.add_argument("-t2","--tr2",help='plot tracer 2 (ejecta)',action='store_true')
	parser.add_argument("-t3","--tr3",help='plot tracer 3 (wind)',action='store_true')
	parser.add_argument("-t4","--tr4",help='plot tracer 4 (cloud)',action='store_true')
	parser.add_argument("-t5","--tr5",help='plot tracer 5 (shock x velocity)',action='store_true')
	parser.add_argument("-t6","--tr6",help='plot tracer 6 (shock y velocity)',action='store_true')
	parser.add_argument("-t7","--tr7",help='plot tracer 7 (shock z velocity)',action='store_true')
	parser.add_argument("-t8","--tr8",help='plot tracer 8 ',action='store_true')
	parser.add_argument("-a","--all",help='plot all the quantities',action='store_true')

	#useful quantities
	parser.add_argument("-lw","--linewidth",help='width of the lines for the profile plot')
	parser.add_argument("-ux","--upperx",help='upper x range')
	parser.add_argument("-lx","--lowerx",help='lower x range')
	parser.add_argument("-uy","--uppery",help='upper y range')
	parser.add_argument("-ly","--lowery",help='lower y range')
	parser.add_argument("-uz","--upperz",help='upper z range')
	parser.add_argument("-lz","--lowerz",help='lower z range')
	parser.add_argument("-ms","--mass",help='ZAMS mass of the star in Msun')
	parser.add_argument("-en","--energy",help='kinetic energy of the explosion')

	args = parser.parse_args()

	######################## error handling in command line parsing ########################

	if ((args.single is False and (args.nfile or args.selstep) is not None) or (args.single is True  and (args.nfile or args.selstep) is None)):
		parser.print_help()
		parser.error('The --single (-s) argument requires the --nfile (-n) argument')
		sys.exit()
		
	if(args.movie==True and args.single==True):
		parser.error('can\'t make the movie in single mode, if you want to make only the movie use the --onlymovie (-om) option')
		sys.exit()

	if(args.movie==True and args.onlymovie==True):
		print("WARNING: The --onlymovie option has been called alongside the --movie one, if you want to make only the movie of existing plots call the --onlymovie option.\n")
		while True:	
			text = input("Do you want to proceed with the --movie option? [y/n]:")
			if(text=='y' or text=='Y' or text=='yes' or text=='Yes' or text=='YES'):
				args.onlymovie=False
				break
			if(text=='n' or text=='N' or text=='no' or text=='No' or text=='NO'):
				args.movie=False
				break
			else:
				print("Incorrect input, repeat\n")

	######################## directories checking ########################
	
	dirpath = os.getcwd()+'/'
	
	# density folder
	if (not os.path.exists(dirpath+'img_rho/')):
		if(args.onlymovie==True and (args.rho==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_rho')

	# density profile folder
	if (not os.path.exists(dirpath+'img_rho_profile/')):
		if(args.onlymovie==True and ((args.rho==True and args.profile==True) or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_rho_profile')

	# pressure folder
	if (not os.path.exists(dirpath+'img_prs/')):
		if(args.onlymovie==True and (args.prs==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_prs')
		
	# temperature folder
	if (not os.path.exists(dirpath+'img_tmp/')):
		if(args.onlymovie==True and (args.tmp==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tmp')

	# temperature profile folder
	if (not os.path.exists(dirpath+'img_tmp_profile/')):
		if(args.onlymovie==True and ((args.tmp==True and args.profile==True) or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tmp_profile')
		
	# x velocity folder
	if (not os.path.exists(dirpath+'img_vx1/')):
		if(args.onlymovie==True and (args.vx1==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_vx1')

	# y velocity folder
	if (not os.path.exists(dirpath+'img_vx2/')):
		if(args.onlymovie==True and (args.vx2==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_vx2')

	# z velocity folder
	if (not os.path.exists(dirpath+'img_vx3/')):
		if(args.onlymovie==True and (args.vx3==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_vx3')
	
	# tracer 1 folder
	if (not os.path.exists(dirpath+'img_tr1/')):
		if(args.onlymovie==True and (args.tr1==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr1')
	
	# tracer 2 folder
	if (not os.path.exists(dirpath+'img_tr2/')):
		if(args.onlymovie==True and (args.tr2==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr2')
	
	# tracer 3 folder
	if (not os.path.exists(dirpath+'img_tr3/')):
		if(args.onlymovie==True and (args.tr3==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr3')
	
	# tracer 4 folder
	if (not os.path.exists(dirpath+'img_tr4/')):
		if(args.onlymovie==True and (args.tr4==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr4')
	
	# tracer 5 folder
	if (not os.path.exists(dirpath+'img_tr5/')):
		if(args.onlymovie==True and (args.tr5==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr5')
	
	# tracer 6 folder
	if (not os.path.exists(dirpath+'img_tr6/')):
		if(args.onlymovie==True and (args.tr6==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr6')
	
	# tracer 7 folder
	if (not os.path.exists(dirpath+'img_tr7/')):
		if(args.onlymovie==True and (args.tr7==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr7')
	
	# tracer 8 folder
	if (not os.path.exists(dirpath+'img_tr8/')):
		if(args.onlymovie==True and (args.tr8==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_tr8')
		
	######################## useful quantities ########################

	line_w = 2.6							#linewidth for profile plots
	if(args.linewidth is not None): line_w = float(args.linewidth)

	MASS = 15							#ZAMS mass of the star
	if(args.mass is not None): MASS = float(args.mass)

	######################## get last stepnumber ########################

	with open('step.file', 'r') as ins:
		aux = []
		for line in ins:
			data = line.split()
			print(data)
			aux.append(int(data[0]))
	iii  = len(aux)-1
	stepnumber = aux[iii]

	######################## box properties ########################

	props = dict(boxstyle='round', facecolor='black', alpha=1.0)	# properties of textbox

	######################## Plotting ########################

	if(args.onlymovie==False):

	######################## Multiple mode ########################

		if(args.single==False):
		
		######################## find minima and maxima over all the files ########################
			print("\n\nFinding minima and maxima...\n")
			
			minrho = np.zeros(stepnumber)
			masrho = np.zeros(stepnumber)
			minprs = np.zeros(stepnumber)
			masprs = np.zeros(stepnumber)
			mintmp = np.zeros(stepnumber)
			mastmp = np.zeros(stepnumber)
			minvel = np.zeros(stepnumber)
			masvel = np.zeros(stepnumber)
			
			for step in range(stepnumber):
				
				st_dirpath = os.getcwd()+'/STEP'+str(step)+'/'		#current working directory
				nlinf = pypl.nlast_info(w_dir=st_dirpath, datatype='flt')
				N = nlinf['nlast']+1

				minrho_i = np.zeros(N)
				masrho_i = np.zeros(N)
				minprs_i = np.zeros(N)
				masprs_i = np.zeros(N)
				mintmp_i = np.zeros(N)
				mastmp_i = np.zeros(N)
				minvel_i = np.zeros(N)
				masvel_i = np.zeros(N)

				for i in tqdm(range(N)):
					with contextlib.redirect_stdout(None):
						D = pp.pload(i, w_dir=st_dirpath, datatype='flt')
					
					tmp = D.prs*up*mu*mp/(2*D.rho*ud*kb)		#compute temperature
					
					minrho_i[i] = D.rho.min()
					masrho_i[i] = D.rho.max()
					minprs_i[i] = D.prs.min()
					masprs_i[i] = D.prs.max()
					mintmp_i[i] = tmp.min()
					mastmp_i[i] = tmp.max()
					minvel_i[i] = D.vx1.min()
					masvel_i[i] = D.vx1.max()
			
				minrho[step] = minrho_i.min()
				masrho[step] = masrho_i.max()
				minprs[step] = minprs_i.min()
				masprs[step] = masprs_i.max()
				mintmp[step] = mintmp_i.min()
				mastmp[step] = mastmp_i.max()
				minvel[step] = minvel_i.min()
				masvel[step] = masvel_i.max()
				
				del minrho_i
				del masrho_i 
				del minprs_i 
				del masprs_i 
				del mintmp_i 
				del mastmp_i 
				del minvel_i 
				del masvel_i 
				
			for step in range(stepnumber):
				
				st_dirpath = os.getcwd()+'/STEP'+str(step)+'/'		#current working directory
				nlinf = pypl.nlast_info(w_dir=st_dirpath, datatype='flt')
				N = nlinf['nlast']+1

				print("\n\nPlotting STEP"+str(step)+"...\n")
				
				######################## Compute temperature ########################
				for i in tqdm(range(N)):
					with contextlib.redirect_stdout(None):
						D = pp.pload(i, w_dir=st_dirpath, datatype='flt')
						
					tmp = D.prs*up*mu*mp/(2*D.rho*ud*kb)		#compute temperature
					
					######################### DENSITY ########################
					if(args.rho==True or args.all==True):
						
						#figure
						fig, ax = plt.subplots(figsize=(12,12))
						ti_str = str('{0:.2f}'.format(D.SimTime*utyrs + ti)) + 'yrs'
						im = ax.pcolormesh(D.x1,D.x3,(D.rho.T[:,int(D.n2/2),:]*ud),cmap='inferno', 
							norm=colors.LogNorm(vmin=minrho.min()*ud, vmax=masrho.max()*ud))
						ax.set_aspect('equal',adjustable='box') 
						
						#labels
						ax.set_xlabel(r'x (pc)',labelpad=20) 
						ax.set_xlabel(r'z (pc)',labelpad=20) 
						plt.title(r'{:.2e}'.format(D.SimTime*utyrs)+r' yrs')
						ax.set_position([0.09, -0.71, 0.6, 2.5])
						ax.text(0.79, 0.95, r'M = '+str(MASS)+r'Msun', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
						
						#colorbar
						cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
						plt.colorbar(im,cax=cax2,orientation='vertical',label=r'Density $g/cm^3$')
						
						#save
						plt.savefig(dirpath+'img_rho/'+'STEP'+str(step)+'_'+str(i)+'.png')
						plt.close()
						
						######################## density profile ########################
						if(args.profile==True):
							
							#figure
							fig = plt.figure(figsize=(16,12))
							plt.plot(D.x1,D.rho[:,int(D.n2/2),int(D.n3/2)],linewidth=line_w,color='white')
							
							#ranges
							plt.ylim(minrho.min()*0.6,masrho.max()*1.4)
							plt.yscale('log')
							
							#labels
							plt.xlabel(r'x (pc)',labelpad=20)
							plt.ylabel(r'Density',labelpad=20)
							plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
							plt.text(0.79, 0.95, r'M = '+str(MASS)+r'Msun', transform=ax.transAxes, fontsize=20,
								verticalalignment='top', bbox=props)
							
							#save
							plt.savefig(dirpath+'img_rho_profile/'+'STEP'+str(step)+'_'+str(i)+'.png')
							plt.close()

		######################## Single Mode ########################
		if(args.single==True):
			
			st_dirpath = os.getcwd()+'/STEP'+str(args.selstep)+'/'		#current working directory
			nlinf = pypl.nlast_info(w_dir=st_dirpath, datatype='flt')
			N = nlinf['nlast']+1

			minrho = np.zeros(N)
			masrho = np.zeros(N)
			minprs = np.zeros(N)
			masprs = np.zeros(N)
			mintmp = np.zeros(N)
			mastmp = np.zeros(N)
			minvel = np.zeros(N)
			masvel = np.zeros(N)
							
			with contextlib.redirect_stdout(None):
				D = pp.pload(int(args.nfile), w_dir=st_dirpath, datatype='flt')
			
			tmp = D.prs*up*mu*mp/(2*D.rho*ud*kb)		#compute temperature
			minrho[0] = D.rho.min()
			masrho[0] = D.rho.max()
			minprs[0] = D.prs.min()
			masprs[0] = D.prs.max()
			mintmp[0] = tmp.min()
			mastmp[0] = tmp.max()
			minvel[0] = D.vx1.min()
			masvel[0] = D.vx1.max()

			######################## DENSITY ########################
			if(args.rho==True or args.all==True):

				#figure
				fig, ax = plt.subplots(figsize=(12,12))
				ti_str = str('{0:.2f}'.format(D.SimTime*utyrs + ti)) + 'yrs'
				im = ax.pcolormesh(D.x1,D.x3,(D.rho.T[:,int(D.n2/2),:]*ud),cmap='inferno', 
					norm=colors.LogNorm(vmin=minrho[0]*ud, vmax=masrho[0]*ud))
				ax.set_aspect('equal',adjustable='box') 
				
				#labels
				ax.set_xlabel(r'x (pc)',labelpad=20) 
				ax.set_xlabel(r'z (pc)',labelpad=20) 
				plt.title(r'{:.2e}'.format(D.SimTime*utyrs)+r' yrs')
				ax.set_position([0.09, -0.71, 0.6, 2.5])
				ax.text(0.79, 0.95, r'M = '+str(MASS)+r'Msun', transform=ax.transAxes, fontsize=20,
				verticalalignment='top', bbox=props)
				
				#colorbar
				cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
				plt.colorbar(im,cax=cax2,orientation='vertical',label=r'Density $g/cm^3$')
















			plt.show()	# Show everything was selected





