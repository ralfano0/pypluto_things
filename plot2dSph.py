# Plot 2D spherical coordinate data
# from PLUTO simulation
# Author: Roberto Alfano
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
		#"text.usetex": True,
		"font.family": "sans-serif",
		"font.sans-serif": "Helvetica",
		"font.size": 18
	})

	######################## constants, fixed variables, unit measures ########################

	mu = 1.2889425/2.		#mean molecular weight			
	ti = 0				#initial time
	ud = 1.67262171e-24		#unit density (g)
	ul = 3.0856775807e18		#unit length (cm)
	uv = 1.e8			#unit velocity (cm/s)
	up = (uv**2)*ud			#unit pressure (dyn/cm^2)
	umag = uv*np.sqrt(4*np.pi*ud)	#unit magnetic field (G)
	utyrs = (ul/uv)*3.17098e-8	#unit time (yrs)
	kb = 1.3806505e-16		#Boltzmann constant in cgs

	######################## command line parsing ########################
	parser = argparse.ArgumentParser(description="plot 2D spherical coordinates map of density [-r/--rho], temperature [-t/--tmp], pressure [-p/--prs] and radial velocity [-v/--vx1]. It is possible to plot profile cuts by specifying the option [-pr/--profile] and, with the option [-m/--movie] or [-om/--onlymovie], one can respectively make the plots and the movie or just the movie on already existing plots. For temperature profile plot it is possible to plot the 1/r^2 dependence by specifying the [-i/--ideal] option. It is possible to make and show a single plot [-s/--single] or to make the plots for all the files in the folder [-a/--all]")

	#algorithm variables
	parser.add_argument("-s","--single",action='store_true', help='single file input, requires -n to be specified')
	parser.add_argument("-pr","--profile",action='store_true', help='enable profile cut plot')
	parser.add_argument("-n","--nfile",help='specify file number to plot in single mode')
	parser.add_argument("-i","--ideal",help='plot ideal profiles in profile plots',action='store_true')
	parser.add_argument("-m","--movie",help='make movies',action='store_true')
	parser.add_argument("-om","--onlymovie",help='makes only the movie from existing plots',action='store_true')

	#physical quantities
	parser.add_argument("-r","--rho",help='plot density',action='store_true')
	parser.add_argument("-p","--prs",help='plot pressure',action='store_true')
	parser.add_argument("-t","--tmp",help='plot temperature',action='store_true')
	parser.add_argument("-v","--vx1",help='plot radial velocity',action='store_true')
	parser.add_argument("-a","--all",help='plot all the quantities',action='store_true')

	#useful quantities
	parser.add_argument("-lw","--linewidth",help='width of the lines for the profile plot')
	parser.add_argument("-ur","--upperrad",help='maximum radius to plot')
	parser.add_argument("-lr","--lowerrad",help='minimum radius to plot')
	parser.add_argument("-ut","--upperth",help='maximum value of theta')
	parser.add_argument("-lt","--lowerth",help='minimum value of theta')
	parser.add_argument("-ub","--uboundary",help='upper value of boundary radius')
	parser.add_argument("-lb","--lboundary",help='lower value of boundary radius')
	parser.add_argument("-ms","--mass",help='ZAMS mass of the star in Msun')
	parser.add_argument("-te","--teff",help='effective temperature of the star')

	args = parser.parse_args()

	######################## error handling in command line parsing ########################

	if ((args.single is False and args.nfile is not None) or (args.single is True  and args.nfile is None)):
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

	dirpath = os.getcwd()+'/'			#current working directory

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
		
	# radial velocity folder
	if (not os.path.exists(dirpath+'img_vx1/')):
		if(args.onlymovie==True and (args.vx1==True or args.all==True)):
			print("WARNING: No plot found, cannot apply --onlymovie option")
			sys.exit()
		os.system('mkdir img_vx1')

	######################## useful quantities ########################

	line_w = 2.6							#linewidth for profile plots
	if(args.linewidth is not None): line_w = float(args.linewidth)

	MAX_R = 6.0							#maximum radius to plot
	if(args.upperrad is not None): MAX_R = float(args.upperrad)

	MIN_R = 0.01							#minimum radius to plot
	if(args.lowerrad is not None): MIN_R = float(args.lowerrad)

	MAX_THETA = 90							#maximum theta value for 2D plots
	if(args.upperth is not None): MAX_THETA = float(args.upperth)

	MIN_THETA = 0							#minimum theta value for 2D plots
	if(args.lowerth is not None): MIN_THETA = float(args.lowerth)

	L_BOUND = 0.01							#innermost internal boundary point
	if(args.lboundary is not None): L_BOUND = float(args.lboundary)

	U_BOUND = 0.08							#outermost internal boundary point
	if(args.uboundary is not None): U_BOUND = float(args.uboundary)

	MASS = 15							#ZAMS mass of the star
	if(args.mass is not None): MASS = float(args.mass)

	TEFF = 3630							#T_eff of the star
	if(args.teff is not None): TEFF = float(args.teff)

	######################## initialize arrays ########################

	nlinf = pypl.nlast_info(w_dir=dirpath)
	N = nlinf['nlast']

	props = dict(boxstyle='round', facecolor='black', alpha=1.0)	# properties of textbox

	minrho = np.zeros(N)
	masrho = np.zeros(N)
	minprs = np.zeros(N)
	masprs = np.zeros(N)
	mintmp = np.zeros(N)
	mastmp = np.zeros(N)
	minvel = np.zeros(N)
	masvel = np.zeros(N)

	######################## Plotting ########################

	if(args.onlymovie==False):

	######################## Multiple mode ########################

		if(args.single==False):
		
		######################## find minima and maxima over all the files ########################
			print("\n\nFinding minima and maxima...\n")
			for i in tqdm(range(N)):
				with contextlib.redirect_stdout(None):
					D = pp.pload(i, w_dir=dirpath)
				
				tmp = D.prs*up*mu*ud/(D.rho*ud*kb)		#compute temperature
				
				minrho[i] = D.rho.min()
				masrho[i] = D.rho.max()
				minprs[i] = D.prs.min()
				masprs[i] = D.prs.max()
				mintmp[i] = tmp.min()
				mastmp[i] = tmp.max()
				minvel[i] = D.vx1.min()
				masvel[i] = D.vx1.max()

			print("\n\nPlotting...\n")
			
			######################## Compute temperature ########################
			for i in tqdm(range(N)):
				with contextlib.redirect_stdout(None):
					D = pp.pload(i, w_dir=dirpath)
					
				tmp = D.prs*up*mu*ud/(D.rho*ud*kb)		#compute temperature
				
				r, th = np.meshgrid(D.x1,D.x2)			# meshgrid for plotting
				
				######################### DENSITY ########################
				if(args.rho==True or args.all==True):
					
					#figure
					fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
					im = ax.pcolormesh(th,r,(D.rho.T*ud),cmap='inferno',
						norm=colors.LogNorm(vmin=(minrho.min()*ud),vmax=(masrho.max()*ud)))
					#ranges
					ax.set_thetamin(MIN_THETA)
					ax.set_thetamax(MAX_THETA)
					ax.set_rmax(MAX_R)
					
					#labels
					ax.set_xlabel(r'r (pc)',labelpad=20) 
					plt.title(r'{:.2e}'.format(D.SimTime*utyrs)+r' yrs')
					ax.set_position([0.09, -0.71, 0.6, 2.5])
					ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
					verticalalignment='top', bbox=props)
					
					#colorbar
					cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
					plt.colorbar(im,cax=cax2,orientation='vertical',label=r'Density $g/cm^3$')
					
					#save
					plt.savefig(dirpath+'img_rho/'+str(i)+'.png')
					plt.close()
					
					######################## density profile ########################
					if(args.profile==True):
						
						#figure
						fig = plt.figure(figsize=(16,12))
						plt.plot(D.x1,D.rho[:,256],linewidth=line_w,color='white',label=r'$\theta\simeq 45^\circ$')	# 45° cut
						plt.plot(D.x1,D.rho[:,0],linewidth=line_w,color='lime',label=r'$\theta\simeq 0^\circ$')		#  0° cut
						
						#ranges
						plt.ylim(minrho.min()*0.6,masrho.max()*1.4)
						plt.yscale('log')
						plt.xlim(MIN_R,MAX_R)
						
						#labels
						plt.legend()
						plt.xlabel(r'r (pc)',labelpad=20)
						plt.ylabel(r'Density',labelpad=20)
						plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
						plt.text(1.54, 1.20, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
						plt.text(0.07, 1.170, r'Boundary \\ $'+str(L_BOUND)+'<r<'+str(U_BOUND)+'$ (pc)', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', color='crimson')
						
						#boundary shading
						r_bound_range = D.x1[(D.x1<U_BOUND) & (D.x1>L_BOUND)]
						plt.fill_between(r_bound_range,minrho.min()*0.5,masrho.max()*1.5,color='crimson',alpha=0.3)
						
						#save
						plt.savefig(dirpath+'img_rho_profile/'+str(i)+'.png')
						plt.close()

				######################## PRESSURE ########################
				if(args.prs==True or args.all==True):
					
					#figure
					fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
					im = ax.pcolormesh(th,r,(D.prs.T*up),cmap='inferno',
							norm=colors.LogNorm(vmin=(minprs.min()*up),vmax=(masprs.max()*up)))
					
					#ranges
					ax.set_thetamin(MIN_THETA)
					ax.set_thetamax(MAX_THETA)
					ax.set_rmax(MAX_R)
					
					#labels
					ax.set_xlabel('r (pc)',labelpad=20)
					plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
					ax.set_position([0.09, -0.71, 0.6, 2.5])
					ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
					
					#colorbar
					cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
					plt.colorbar(im,cax=cax2,orientation='vertical',label='Pressure $dyn/cm^2$')
					
					#save
					plt.savefig(dirpath+'img_prs/'+str(i)+'.png')
					plt.close()

				######################## TEMPERTURE ########################
				if(args.tmp==True or args.all==True):
				
					#figure
					fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
					im = ax.pcolormesh(th,r,(tmp.T),cmap='inferno',
							norm=colors.LogNorm(vmin=(mintmp.min()),vmax=(mastmp.max())))
							
					#ranges
					ax.set_thetamin(MIN_THETA)
					ax.set_thetamax(MAX_THETA)
					ax.set_rmax(MAX_R)
					
					#labels
					ax.set_xlabel('r (pc)',labelpad=20)
					plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
					ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
					ax.set_position([0.09, -0.71, 0.6, 2.5])
					
					#colorbar
					cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
					plt.colorbar(im,cax=cax2,orientation='vertical',label='Temperature $K$')
					
					#save
					plt.savefig(dirpath+'img_tmp/'+str(i)+'.png')
					plt.close()
					
					######################## temperature profile ########################
					if(args.profile==True):
					
						#figure
						fig = plt.figure(figsize=(16,12))
						plt.plot(D.x1,tmp[:,256],linewidth=line_w,color='white',label=r'$\theta\simeq 45^\circ$')	# 45° cut
						plt.plot(D.x1,tmp[:,0],linewidth=line_w,color='lime',label=r'$\theta\simeq 0^\circ$')		#  0° cut
						
						#ideal option
						if(args.ideal==True):	
							plt.plot(D.x1,(TEFF*U_BOUND**2)/(D.x1**2),linewidth=line_w,color='red',label=r'$1/r^2$')
							
						#ranges
						plt.ylim(mintmp.min()*0.6,mastmp.max()*1.4)
						plt.yscale('log')
						plt.xlim(MIN_R,MAX_R)
						
						#labels
						plt.legend()
						plt.xlabel(r'r (pc)',labelpad=20)
						plt.ylabel(r'Temperature K',labelpad=20)
						plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
						plt.text(1.54, 1.20, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
							verticalalignment='top', bbox=props)
						plt.text(0.07, 1.170, r'Boundary \\ $'+str(L_BOUND)+'<r<'+str(U_BOUND)+'$ (pc)', transform=ax.transAxes, fontsize=20,
							verticalalignment='top', color='crimson')
						
						#boundary shading
						r_bound_range = D.x1[(D.x1<U_BOUND) & (D.x1>L_BOUND)]
						plt.fill_between(r_bound_range,mintmp.min()*0.5,mastmp.max()*1.5,color='crimson',alpha=0.3)
						
						#save
						plt.savefig(dirpath+'img_tmp_profile/'+str(i)+'.png')
						plt.close()

				######################## RADIAL VELOCITY ########################
				if(args.vx1==True or args.all==True):
				
					#figure
					fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(12,12))
					im = ax.pcolormesh(th,r,(D.vx1.T*uv),cmap='inferno',
							norm=colors.Normalize(vmin=(minvel.min()*uv),vmax=(masvel.max()*uv)))
							
					#ranges
					ax.set_thetamin(MIN_THETA)
					ax.set_thetamax(MAX_THETA)
					ax.set_rmax(MAX_R)
					
					#labels
					ax.set_xlabel('r (pc)',labelpad=20)
					plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
					ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
							verticalalignment='top', bbox=props)
					ax.set_position([0.09, -0.71, 0.6, 2.5])
					
					#colorbar
					cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
					plt.colorbar(im,cax=cax2,orientation='vertical',label='Radial Velocity $cm/s$')
					
					#save
					plt.savefig(dirpath+'img_vx1/'+str(i)+'.png')
					plt.close()


			print("\n\nFinished.\n")
			
		######################## Single Mode ########################
		if(args.single==True):
		
			with contextlib.redirect_stdout(None):
				D = pp.pload(int(args.nfile), w_dir=dirpath)
			
			tmp = D.prs*up*mu*ud/(D.rho*ud*kb)		#compute temperature
			minrho[0] = D.rho.min()
			masrho[0] = D.rho.max()
			minprs[0] = D.prs.min()
			masprs[0] = D.prs.max()
			mintmp[0] = tmp.min()
			mastmp[0] = tmp.max()
			minvel[0] = D.vx1.min()
			masvel[0] = D.vx1.max()
			
			r, th = np.meshgrid(D.x1,D.x2)

			######################## DENSITY ########################
			if(args.rho==True or args.all==True):
				
				#figure
				fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(10,10))
				im = ax.pcolormesh(th,r,(D.rho.T*ud),cmap='inferno',
					norm=colors.LogNorm(vmin=(minrho[0]*ud),vmax=(masrho[0]*ud)))
					
				#ranges
				ax.set_thetamin(MIN_THETA)
				ax.set_thetamax(MAX_THETA)
				ax.set_rmax(MAX_R)
				ax.set_xlabel(r'r (pc)',labelpad=20) 
				
				#labels
				plt.title(r'{:.2e}'.format(D.SimTime*utyrs)+r' yrs')
				ax.set_position([0.09, -0.71, 0.6, 2.5])
				ax.text(0.79, 0.95, r'M = '+str(MASS)+r'Msun T = '+str(TEFF)+r'K', transform=ax.transAxes, fontsize=20,
				verticalalignment='top', bbox=props)
				
				#colorbar
				cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
				plt.colorbar(im,cax=cax2,orientation='vertical',label=r'Density g/cm^3')
				
				######################## density profile ########################
				if(args.profile==True):
				
					#figure
					fig2 = plt.figure(figsize=(13,10))
					plt.plot(D.x1,D.rho[:,256],linewidth=line_w,color='white',label=r'$\theta\simeq 45^\circ$')	# 45° cut
					plt.plot(D.x1,D.rho[:,0],linewidth=line_w,color='lime',label=r'$\theta\simeq 0^\circ$')		#  0° cut
					
					#ranges
					plt.ylim(minrho[0]*0.6,masrho[0]*1.4)
					plt.yscale('log')
					plt.xlim(MIN_R,MAX_R)
					
					#labels
					plt.legend()
					plt.xlabel(r'r (pc)',labelpad=20)
					plt.ylabel(r'Density',labelpad=20)
					plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
					plt.text(1.54, 1.20, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
					verticalalignment='top', bbox=props)
					plt.text(0.07, 1.170, r'Boundary \\ $'+str(L_BOUND)+'<r<'+str(U_BOUND)+'$ (pc)', transform=ax.transAxes, fontsize=20,
					verticalalignment='top', color='crimson')
					
					#boundary shading
					r_bound_range = D.x1[(D.x1<U_BOUND) & (D.x1>L_BOUND)]
					plt.fill_between(r_bound_range,minrho[0]*0.5,masrho[0]*1.5,color='crimson',alpha=0.3)

			######################## PRESSURE ########################
			if(args.prs==True or args.all==True):
			
				#figure
				fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(10,10))
				im = ax.pcolormesh(th,r,(D.prs.T*up),cmap='inferno',
						norm=colors.LogNorm(vmin=(minprs[0]*up),vmax=(masprs[0]*up)))
				
				#ranges
				ax.set_thetamin(MIN_THETA)
				ax.set_thetamax(MAX_THETA)
				ax.set_rmax(MAX_R)
				
				#labels
				ax.set_xlabel('r (pc)',labelpad=20)
				plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
				ax.set_position([0.09, -0.71, 0.6, 2.5])
				ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
					verticalalignment='top', bbox=props)
					
				#colors
				cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
				plt.colorbar(im,cax=cax2,orientation='vertical',label='Pressure $dyn/cm^2$')

			######################## TEMPERATURE ########################
			if(args.tmp==True or args.all==True):
			
				#figure
				fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(10,10))
				im = ax.pcolormesh(th,r,(tmp.T),cmap='inferno',
						norm=colors.LogNorm(vmin=(mintmp[0]),vmax=(mastmp[0])))
				
				#ranges
				ax.set_thetamin(MIN_THETA)
				ax.set_thetamax(MAX_THETA)
				ax.set_rmax(MAX_R)
				
				#labels
				ax.set_xlabel('r (pc)',labelpad=20)
				plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
				ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
					verticalalignment='top', bbox=props)
				ax.set_position([0.09, -0.71, 0.6, 2.5])
				
				#colorbar
				cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
				plt.colorbar(im,cax=cax2,orientation='vertical',label='Temperature $K$')
				
				######################## temperature profile ########################
				if(args.profile==True):
				
					#figure
					fig = plt.figure(figsize=(13,10))
					plt.plot(D.x1,tmp[:,256],linewidth=line_w,color='white',label=r'$\theta\simeq 45^\circ$')	# 45° cut
					plt.plot(D.x1,tmp[:,0],linewidth=line_w,color='lime',label=r'$\theta\simeq 0^\circ$')		#  0° cut
					
					#ideal option
					if(args.ideal==True):	
						plt.plot(D.x1,(TEFF*U_BOUND**2)/(D.x1**2),linewidth=line_w,color='red',label=r'$1/r^2$')
						
					#ranges
					plt.ylim(mintmp[0]*0.6,mastmp[0]*1.4)
					plt.yscale('log')
					plt.xlim(MIN_R,MAX_R)
					
					#label
					plt.legend()
					plt.xlabel(r'r (pc)',labelpad=20)
					plt.ylabel(r'Temperature K',labelpad=20)
					plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
					plt.text(1.54, 1.20, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
					plt.text(0.07, 1.170, r'Boundary \\ $'+str(L_BOUND)+'<r<'+str(U_BOUND)+'$ (pc)', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', color='crimson')
						
					#boundary shading
					r_bound_range = D.x1[(D.x1<U_BOUND) & (D.x1>L_BOUND)]
					plt.fill_between(r_bound_range,mintmp[0]*0.5,mastmp[0]*1.5,color='crimson',alpha=0.3)

			######################## RADIAL VELOCITY ########################
			if(args.vx1==True or args.all==True):
			
				#figure
				fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),figsize=(10,10))
				im = ax.pcolormesh(th,r,(D.vx1.T*uv),cmap='inferno',
						norm=colors.Normalize(vmin=(minvel[0]*uv),vmax=(masvel[0]*uv)))
				
				#ranges
				ax.set_thetamin(MIN_THETA)
				ax.set_thetamax(MAX_THETA)
				ax.set_rmax(MAX_R)
				
				#labels
				ax.set_xlabel('r (pc)',labelpad=20)
				plt.title('{:.2e}'.format(D.SimTime*utyrs)+' yrs')
				ax.text(0.79, 0.95, r'$$M = '+str(MASS)+r'M_\odot$$ $$T = '+str(TEFF)+r'K$$', transform=ax.transAxes, fontsize=20,
						verticalalignment='top', bbox=props)
				ax.set_position([0.09, -0.71, 0.6, 2.5])
				
				#colorbar
				cax2 = fig.add_axes([0.85,0.05,0.03,0.86])
				plt.colorbar(im,cax=cax2,orientation='vertical',label='Radial Velocity $cm/s$')
				
			plt.show()	# Show everything was selected
			
			
	######################## Animating ########################
	if(args.movie==True or args.onlymovie==True):
		print("Animating...\n")
		
		# density		
		if (os.path.exists(dirpath+'img_rho/') and args.rho):
			os.system('rm img_rho/out_rho.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_rho/%d.png' img_rho/out_rho.mp4")
			
		# density profile	
		if (os.path.exists(dirpath+'img_rho_profile/') and (args.rho and args.profile)):
			os.system('rm img_rho_profile/out_rho_profile.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_rho_profile/%d.png' img_tmp_profile/out_rho_profile.mp4")
			
		# pressure
		if (os.path.exists(dirpath+'img_prs/') and args.prs):
			os.system('rm img_prs/out_prs.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_prs/%d.png' img_prs/out_prs.mp4")
			
		# temperature	
		if (os.path.exists(dirpath+'img_tmp/') and args.tmp):
			os.system('rm img_tmp/out_tmp.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_tmp/%d.png' img_tmp/out_tmp.mp4")
			
		# temperature profile	
		if (os.path.exists(dirpath+'img_tmp_profile/') and (args.tmp and args.profile)):
			os.system('rm img_tmp_profile/out_tmp_profile.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_tmp_profile/%d.png' img_tmp_profile/out_rho_profile.mp4")
			
		# radial velocity	
		if (os.path.exists(dirpath+'img_vx1/') and args.vx1):
			os.system('rm img_vx1/out_vx1.mp4')
			os.popen("ffmpeg -framerate 30 -i 'img_vx1/%d.png' img_vx1/out_vx1.mp4")
		
		print("Finished!\n")
