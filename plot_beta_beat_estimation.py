import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pickle
import tfs
import json

FONTSIZE = 16

COLOR_X = u'#1f77b4'
COLOR_Y = u'#ff7f0e'

"""
B1_CORRECTOR_PATH copied from:
/omc3/model/accelerators/lhc/2012/correctors/correctors_b1/beta_correctors.json
"""	
B1_CORRECTOR_PATH = "beta_correctors.json" 


	
def plot_correction_correlation(correction_dict_pickle, outputfile_dir, variable_categories, proton, seed = "0", opticsfile_comparison="opticsfile.180"):
	with open(f"{outputfile_dir}{correction_dict_pickle}","rb") as p:
                correction_dict_seed = pickle.load(p)
	
	strength_dict = {}
	for j , key2 in enumerate(correction_dict_seed[seed]):
		strength_dict[key2] = _get_strength_dict(correction_dict_seed[seed][key2])
		
	
	with open(B1_CORRECTOR_PATH) as f:
    		data = f.read()
	families = json.loads(data)
	
	beta_star_l = np.array([100*_get_beta_star(proton, opticsfile) for opticsfile in strength_dict.keys()])
	
	color_dict = {"MQM" : COLOR_X , "MQY" : COLOR_Y}
	
	fig , ax = plt.subplots()
	for variable_category in variable_categories:
		for i , corrector in enumerate(families[variable_category]):
			corrector_strength_dict = {}		
			for j , opticsfile in enumerate(strength_dict):	
				corrector_strength_dict[opticsfile] = strength_dict[opticsfile][corrector]
			if i == 0:
				ax.plot(corrector_strength_dict["opticsfile.120"] / 1e-5 , (corrector_strength_dict[opticsfile_comparison] - corrector_strength_dict["opticsfile.120"] ) / 1e-5, "x" , color = color_dict[variable_category],label = variable_category)
			else:	
				ax.plot(corrector_strength_dict["opticsfile.120"] / 1e-5 , (corrector_strength_dict[opticsfile_comparison] - corrector_strength_dict["opticsfile.120"] ) / 1e-5, "x" , color = color_dict[variable_category])


	ax.legend(fontsize=20)
	
	unit = "$\,[10^{-5} \, m^{-2}]$"
	beta_star = f"{int(100*_get_beta_star(proton, opticsfile_comparison))}cm"
	ax.set_xlabel(f"$K_1^{{60cm}}${unit}", fontsize = FONTSIZE)
	ax.set_ylabel(fr"$\left(K_1^{{{beta_star}}} - K_1^{{60cm}}\right)${unit}",fontsize = FONTSIZE)
	ax.set_xlim(-5,5)
	ax.set_ylim(-5,5)
	ax.tick_params(axis="x",labelsize=16)       	
	ax.tick_params(axis="y",labelsize=16)
	plt.tight_layout()
	plt.savefig(f"plots/correction_correlation_{opticsfile_comparison}.pdf")
	plt.show()
	
def _get_strength_dict(correction):
	strength_dict = {}
	for line in correction.splitlines():
			split = line.split("=")
			name = split[0].replace(" ","")
			number = split[1].replace(split[0],"")
			number = number.replace(" ","")
			number = float(number.replace(";",""))
			
			if name not in strength_dict:
				strength_dict[name] = number	
			else:
				strength_dict[name]+= number			
	return strength_dict			
		
			
def plot_correction_ramp(opticsfiles, opticsfiles_correction, savename, proton):
	correction_weights = {opticsfile : np.zeros(len(opticsfiles)) for opticsfile in opticsfiles_correction}
	beta_star_l = []
	for i , opticsfile in enumerate(opticsfiles):
		beta_star_l.append(_get_beta_star(proton, opticsfile))
	
		weight_dict = _get_weight_dict(opticsfile, opticsfiles_correction)
		for key in weight_dict:
			correction_weights[key][i] = weight_dict[key]
	
	beta_star_l = np.array(beta_star_l)
	
	fig , ax = plt.subplots()
	for correction in np.flip(opticsfiles_correction):
		ax.plot(beta_star_l * 100, correction_weights[correction] * 100, label = fr"$\beta^* $ = {_get_beta_star(proton, correction)*100}")
	
	for optics in opticsfiles_correction:
		beta_star_correction = _get_beta_star(proton, optics)*100
		ax.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color = "black")
	
	ax.set_xlabel(r"$\beta^*\,$[cm]", fontsize=FONTSIZE)
	ax.set_ylabel(r"correction weight [%]", fontsize=FONTSIZE)
	ax.legend(loc='lower center',fontsize=12, bbox_to_anchor=(0.5, 1.03),
          fancybox=True, shadow=False, ncol=5)
	plt.tight_layout()
	plt.savefig(f"plots/{savename}")
	plt.show()
		
		

def _get_weight_dict(opticsfile, opticsfiles_correction):
	optics_number = _optics_number(opticsfile)
	#checking if the opticsfile is first of last
	
	if optics_number <= _optics_number(opticsfiles_correction[0]):
		print("<=",optics_number)
		return {opticsfiles_correction[0] : 1}
	elif optics_number >= _optics_number(opticsfiles_correction[len(opticsfiles_correction)-1]):
		print(">=",optics_number)
		return {opticsfiles_correction[len(opticsfiles_correction)-1] : 1}
		return
	
	#if opticsfile is in list or inbetween
	for i in range(1,len(opticsfiles_correction)):
		optics_number_correction1 = _optics_number(opticsfiles_correction[i - 1])
		optics_number_correction2 = _optics_number(opticsfiles_correction[i])
		if optics_number_correction1 == optics_number:
			print("==",optics_number)
			return {opticsfiles_correction[i-1] : 1}
		elif optics_number < optics_number_correction2:
			print("elif",optics_number)
			weight2 = (optics_number - optics_number_correction1) / (optics_number_correction2 - optics_number_correction1)
			print(weight2)
			weight1 = 1 - weight2
			return  {opticsfiles_correction[i-1] : weight1, opticsfiles_correction[i] : weight2}
	
def _optics_number(opticsfile):
	return int(opticsfile.split(".")[1])


def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))

def plot_magnet_strength(correction_dict_pickle, outputfile_dir, opticsfile, variable_categories, correction_path, method="max"):
	with open(f"{outputfile_dir}{correction_dict_pickle}","rb") as p:
                correction_dict_seed = pickle.load(p)
	
	strength_dict = {}
	for i , key1 in enumerate(correction_dict_seed):
		for j , key2 in enumerate(correction_dict_seed[key1]):
			if i == 0:
				strength_dict[key2] = {}	
				
			for line in correction_dict_seed[key1][key2].splitlines():
				split = line.split("=")
				name = split[0].replace(" ","")
				number = split[1].replace(split[0],"")
				number = float(number.replace(";",""))
				
				if i == 0:
					strength_dict[key2][name] = np.zeros(len(correction_dict_seed))
					strength_dict[key2][name][i] += number	
				else:
					strength_dict[key2][name][i] += number	
	
	
	with open(correction_path,"r") as f:
		correction = f.read()
	strength_dict_real = _get_strength_dict(correction)
	
	with open(B1_CORRECTOR_PATH) as f:
    		data = f.read()
	families = json.loads(data)
	
	#for key in trength_dict[opticsfile]:
	
	method_dict = {
	"rms" : lambda x : rms(x,axis=0),
	"max" : lambda x : np.max(abs(x),axis=0),
	}
	color_l = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728']
	fig , ax = plt.subplots()
	for i , category in enumerate(variable_categories):
		N_family = len(families[category])
		N_seed = len(correction_dict_seed)
		data = np.zeros((N_family,N_seed))
		data_real =[]
		for j , corrector in enumerate(families[category]):
			#print(strength_dict[opticsfile][corrector])
			data[j,:] = strength_dict[opticsfile][corrector]
			if corrector in strength_dict_real:
				data_real.append(strength_dict_real[corrector])
				
			del strength_dict[opticsfile][corrector]
			
		plot_list = method_dict[method](data)
		plot_list_real = method_dict[method](np.array(data_real)) 
		print(plot_list_real)
		if method == "rms":
			b = 6e-5
			scale = 5
		elif method == "max":
			b = 20.0e-5	
			scale = 5
			
		
		
		ax.hist(plot_list * (10 ** scale), alpha=0.5,bins=50,range = (0,b* 10 ** scale), color = color_l[i], edgecolor = "black", label=category)
		ax.hist(plot_list_real* (10 ** scale),alpha=1,bins=50,range = (0,b* 10 ** scale), color = color_l[i], edgecolor = "red")
		
		arrow = mpatches.FancyArrowPatch((plot_list_real* (10 ** scale),5),(plot_list_real* (10 ** scale),1),mutation_scale=13)
		ax.add_patch(arrow)
		
		
	unit = "$\,[10^{-5} \, m^{-2}]$"
	ax.set_xlabel(f"{method}{{$K_1$}}{unit}",fontsize = 25)
	ax.set_ylabel("seeds",fontsize = 25)
	ax.tick_params(axis="x",labelsize=16)       	
	ax.tick_params(axis="y",labelsize=16)
	ax.legend(fontsize = 20)
	plt.tight_layout()
	plt.savefig(f"plots/{method}_strength_hist_{opticsfile}.pdf")
	plt.show()


def plot_beta_beat(outputfile_dir, pickle_name, model_dir, savename, plane = "X"):
	with open(f"{outputfile_dir}{pickle_name}","rb") as p:
                multiple_correction_dict = pickle.load(p)
	
	model_df = tfs.read(f"{model_dir}twiss_elements.dat",index="NAME")
	
	IP_BPMs = ["BPMSW.1L5.B1","BPMSW.1R5.B1","BPMSW.1L1.B1","BPMSW.1R1.B1"]
	
	fig , ax = plt.subplots()
	for key in multiple_correction_dict:
		df = multiple_correction_dict[key] #.loc[IP_BPMs,:]
		betx_model = model_df.loc[list(df.index),f"BET{plane}"].to_numpy()
		bb= (df[f"BET{plane}"].to_numpy() - betx_model) / betx_model
		
		print((df[f"BET{plane}"] - model_df.loc[list(df.index),:][f"BET{plane}"]) / model_df.loc[list(df.index),:][f"BET{plane}"])
		
		#rms_bb = np.sqrt(np.mean(bb**2)) * 100
		rms_bb = _get_beta_beat(df,model_df.loc[list(df.index),:],plane)
		ax.plot(df["S"],bb,label = f"RMS[%]:{round(rms_bb,1)}")
	
	ax.legend(fontsize  = FONTSIZE)
	ax.set_xlabel("position [m]",fontsize  = FONTSIZE)
	ax.set_ylabel(r"$\Delta \beta_x$",fontsize  = FONTSIZE)
	plt.savefig(f"plots/{savename}")
	plt.show()
	

def plot_optics_change(outputfile_dir, pickle_name, savename, proton, opticsfiles_correction = [], method = "rms"):
	with open(f"{outputfile_dir}{pickle_name}","rb") as p:
		seed_dict = pickle.load(p)
	
	beta_star_l = []
	betx_beat_rms_l = []
	bety_beat_rms_l = [] 

	for i , key in enumerate(seed_dict):
		twiss_dict = seed_dict[key]
		
		
		for j , key in enumerate(twiss_dict):
			model_df = twiss_dict[key]["model"]
			df = twiss_dict[key]["meas"]
			
			if i == 0:
				betx_beat_rms_l.append([_get_beta_beat(df,model_df,"X",method = method)])
				bety_beat_rms_l.append([_get_beta_beat(df,model_df,"Y",method = method)])
				beta_star_l.append(_get_beta_star(proton, key))
			else:
				betx_beat_rms_l[j].append(_get_beta_beat(df,model_df,"X",method = method))
				bety_beat_rms_l[j].append(_get_beta_beat(df,model_df,"Y",method = method))
	
	#converting to numpy array
	beta_star_l = np.array(beta_star_l) * 100
	betx_beat_rms_l = np.array(betx_beat_rms_l)
	bety_beat_rms_l = np.array(bety_beat_rms_l)
	
	#average
	betx_beat_rms_avg_l = np.mean(betx_beat_rms_l,axis=1)
	bety_beat_rms_avg_l = np.mean(bety_beat_rms_l,axis=1)
	
	#standard deviation
	betx_beat_rms_std_l = np.std(betx_beat_rms_l,axis=1)
	bety_beat_rms_std_l = np.std(bety_beat_rms_l,axis=1)
	
	#max
	betx_beat_rms_max_l = np.max(betx_beat_rms_l,axis=1)
	bety_beat_rms_max_l = np.max(bety_beat_rms_l,axis=1)
			
	fig , ax = plt.subplots()
	
	#betx
	ax.errorbar(beta_star_l, betx_beat_rms_avg_l, yerr=betx_beat_rms_std_l, label=r"$\beta_x$", marker="+", markersize=8, color = COLOR_X)
	ax.plot(beta_star_l, betx_beat_rms_max_l,"x", label= r"max $\beta_x$", color = COLOR_X)
	
	#bety
	ax.errorbar(beta_star_l, bety_beat_rms_avg_l, yerr=bety_beat_rms_std_l, label=r"$\beta_y$", marker="+", markersize=8, color = COLOR_Y)
	ax.plot(beta_star_l, bety_beat_rms_max_l,"x", label= r"max $\beta_y$", color = COLOR_Y)
	
	for optics in opticsfiles_correction:
		beta_star_correction = _get_beta_star(proton, optics)*100
		ax.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color = "black")
	
	ax.legend(fontsize = FONTSIZE)
	ax.set_xlabel(r"$\beta^*$[cm]",fontsize = FONTSIZE)
	ax.set_ylabel(fr"{method}($\Delta\beta$)[%]",fontsize = FONTSIZE)
	ax.tick_params(axis="x",labelsize=16)       	
	ax.tick_params(axis="y",labelsize=16)
	plt.tight_layout()
	plt.savefig(f"plots/{savename}")
	plt.show()
	
	
def plot_IP_optics_change(outputfile_dir, pickle_name, savename, proton, opticsfiles_correction = []):
	with open(f"{outputfile_dir}{pickle_name}","rb") as p:
		seed_dict = pickle.load(p)
	
	beta_star_l = []
	betx_beat_IP1_l = []
	bety_beat_IP1_l = []
	betx_beat_IP5_l = []
	bety_beat_IP5_l = []
	
	IP_BPMs = ["BPMSW.1L5.B1","BPMSW.1R5.B1","BPMSW.1L1.B1","BPMSW.1R1.B1"]
	
	for i , key in enumerate(seed_dict):
		twiss_dict = seed_dict[key]
		
		print(key)
		for j , key in enumerate(twiss_dict):
			model_df = twiss_dict[key]["model"]
			df = twiss_dict[key]["meas"]
			
			if i == 0:
				betx_beat_IP1_l.append([_get_IP_beta_beat(df,model_df,"X","IP1")])
				bety_beat_IP1_l.append([_get_IP_beta_beat(df,model_df,"Y","IP1")])
				betx_beat_IP5_l.append([_get_IP_beta_beat(df,model_df,"X","IP5")])
				bety_beat_IP5_l.append([_get_IP_beta_beat(df,model_df,"Y","IP5")])
				beta_star_l.append(_get_beta_star(proton, key))
			else:
				betx_beat_IP1_l[j].append(_get_IP_beta_beat(df,model_df,"X","IP1"))
				bety_beat_IP1_l[j].append(_get_IP_beta_beat(df,model_df,"Y","IP1"))
				betx_beat_IP5_l[j].append(_get_IP_beta_beat(df,model_df,"X","IP5"))
				bety_beat_IP5_l[j].append(_get_IP_beta_beat(df,model_df,"Y","IP5"))
	
	#converting to numpy array
	beta_star_l = np.array(beta_star_l) * 100
	betx_beat_IP1_l = np.array(betx_beat_IP1_l)
	bety_beat_IP1_l = np.array(bety_beat_IP1_l)
	betx_beat_IP5_l = np.array(betx_beat_IP5_l)
	bety_beat_IP5_l = np.array(bety_beat_IP5_l)
	
	print("BETX IP1",betx_beat_IP1_l)
	print("BETY IP1",bety_beat_IP1_l)
	print("BETX IP5",betx_beat_IP5_l)
	print("BETY IP5",bety_beat_IP5_l)
	
	#average
	betx_beat_IP1_avg_l = np.mean(betx_beat_IP1_l,axis=1)
	bety_beat_IP1_avg_l = np.mean(bety_beat_IP1_l,axis=1)
	betx_beat_IP5_avg_l = np.mean(betx_beat_IP5_l,axis=1)
	bety_beat_IP5_avg_l = np.mean(bety_beat_IP5_l,axis=1)
	
	#standard deviation
	betx_beat_IP1_std_l = np.std(betx_beat_IP1_l,axis=1)
	bety_beat_IP1_std_l = np.std(bety_beat_IP1_l,axis=1)
	betx_beat_IP5_std_l = np.std(betx_beat_IP5_l,axis=1)
	bety_beat_IP5_std_l = np.std(bety_beat_IP5_l,axis=1)
	
	#max
	betx_beat_IP1_max_l = absmaxND(betx_beat_IP1_l,axis=1)
	bety_beat_IP1_max_l = absmaxND(bety_beat_IP1_l,axis=1)
	betx_beat_IP5_max_l = absmaxND(betx_beat_IP5_l,axis=1)
	bety_beat_IP5_max_l = absmaxND(bety_beat_IP5_l,axis=1)
		
	fig , (ax1,ax2) = plt.subplots(2,sharex=True)
	plt.setp(ax1.get_xticklabels(), visible=False)
	
	ax1.errorbar(beta_star_l, betx_beat_IP1_avg_l,yerr=betx_beat_IP1_std_l, marker = "+", label = r"$\beta_x$",color=COLOR_X)
	ax1.errorbar(beta_star_l, bety_beat_IP1_avg_l,yerr=bety_beat_IP1_std_l, marker = "+", label = r"$\beta_y$",color=COLOR_Y)
	ax1.plot(beta_star_l, betx_beat_IP1_max_l,"x", label = fr"max $\beta_x$",color=COLOR_X)
	ax1.plot(beta_star_l, bety_beat_IP1_max_l,"x", label = fr"max $\beta_y$",color=COLOR_Y)
	
	ax2.errorbar(beta_star_l, betx_beat_IP5_avg_l,yerr=betx_beat_IP5_std_l, marker = "+", label = r"$\beta_x^{IP5}$",color=COLOR_X)
	ax2.errorbar(beta_star_l, bety_beat_IP5_avg_l,yerr=bety_beat_IP5_std_l, marker = "+", label = r"$\beta_y^{IP5}$",color=COLOR_Y)
	ax2.plot(beta_star_l, betx_beat_IP5_max_l,"x", label = fr"max $\beta_x$",color=COLOR_X)
	ax2.plot(beta_star_l, bety_beat_IP5_max_l,"x", label = fr"max $\beta_y$",color=COLOR_Y)
	
	ax1.tick_params(axis="x",labelsize=13)       	
	ax1.tick_params(axis="y",labelsize=13)
	ax2.tick_params(axis="x",labelsize=13)       	
	ax2.tick_params(axis="y",labelsize=13)
	
	ax1.legend(loc='lower center',fontsize=12, bbox_to_anchor=(0.5, 1.03),
          fancybox=True, shadow=False, ncol=5)
	#ax1.set_xlabel(r"$\beta^*$[cm]",fontsize = FONTSIZE)
	ax1.set_ylabel(r"$\Delta\beta^{IP1}[\%]$",fontsize = FONTSIZE)

	
	#ax2.legend(fontsize = FONTSIZE)
	ax2.set_xlabel(r"$\beta^*$[cm]",fontsize = FONTSIZE)
	ax2.set_ylabel(r"$\Delta\beta^{IP5}[\%]$",fontsize = FONTSIZE)
	print( max([ max(abs(betx_beat_IP1_avg_l)), max(abs(bety_beat_IP1_avg_l)), max(abs(betx_beat_IP1_avg_l)), max(abs(betx_beat_IP1_avg_l))]))
	print( max([ max(abs(betx_beat_IP1_std_l)), max(abs(betx_beat_IP1_std_l)), max(abs(betx_beat_IP1_std_l)), max(abs(betx_beat_IP1_std_l))]))
	
	ylim1 = 1 * max([max(abs(betx_beat_IP1_max_l)),max(abs(bety_beat_IP1_max_l))])
	ylim5 = 1 * max([max(abs(betx_beat_IP5_max_l)),max(abs(bety_beat_IP5_max_l))])
	#ax1.set_ylim(-ylim1,ylim1)
	#ax2.set_ylim(-ylim5,ylim5)
	
	for optics in opticsfiles_correction:
		beta_star_correction = _get_beta_star(proton, optics)*100
		ax1.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color="black")
		ax2.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color="black")
		
	plt.tight_layout()
	plt.savefig(f"plots/{savename}")
	plt.show()

def plot_IP8_optics_change(outputfile_dir, pickle_name, savename, proton, opticsfiles_correction = []):
	with open(f"{outputfile_dir}{pickle_name}","rb") as p:
		seed_dict = pickle.load(p)
	
	beta_star_l = []
	betx_beat_IP8_l = []
	bety_beat_IP8_l = []
	IP_BPMs = ["BPMSW.1L5.B1","BPMSW.1R5.B1","BPMSW.1L1.B1","BPMSW.1R1.B1"]
	
	for i , key in enumerate(seed_dict):
		twiss_dict = seed_dict[key]
		
		print(key)
		for j , key in enumerate(twiss_dict):
			model_df = twiss_dict[key]["model"]
			df = twiss_dict[key]["meas"]
			
			if i == 0:
				betx_beat_IP8_l.append([_get_IP_beta_beat(df,model_df,"X","IP8")])
				bety_beat_IP8_l.append([_get_IP_beta_beat(df,model_df,"Y","IP8")])
				beta_star_l.append(_get_beta_star(proton, key))
			else:
				betx_beat_IP8_l[j].append(_get_IP_beta_beat(df,model_df,"X","IP8"))
				bety_beat_IP8_l[j].append(_get_IP_beta_beat(df,model_df,"Y","IP8"))
	
	#converting to numpy array
	beta_star_l = np.array(beta_star_l) * 100
	betx_beat_IP8_l = np.array(betx_beat_IP8_l)
	bety_beat_IP8_l = np.array(bety_beat_IP8_l)
	
	#average
	betx_beat_IP8_avg_l = np.mean(betx_beat_IP8_l,axis=1)
	bety_beat_IP8_avg_l = np.mean(bety_beat_IP8_l,axis=1)
	
	#standard deviation
	betx_beat_IP8_std_l = np.std(betx_beat_IP8_l,axis=1)
	bety_beat_IP8_std_l = np.std(bety_beat_IP8_l,axis=1)

	#max
	betx_beat_IP8_max_l = absmaxND(betx_beat_IP8_l,axis=1)
	bety_beat_IP8_max_l = absmaxND(bety_beat_IP8_l,axis=1)
	
	fig , ax = plt.subplots()
	ax.errorbar(beta_star_l, betx_beat_IP8_avg_l,yerr=betx_beat_IP8_std_l, marker = "+", label = r"$\beta_x$",color=COLOR_X)
	ax.errorbar(beta_star_l, bety_beat_IP8_avg_l,yerr=bety_beat_IP8_std_l, marker = "+", label = r"$\beta_y$",color=COLOR_Y)
	
	ax.plot(beta_star_l, betx_beat_IP8_max_l,"x", label = fr"max $\beta_x$",color=COLOR_X)
	ax.plot(beta_star_l, bety_beat_IP8_max_l,"x", label = fr"max $\beta_y$",color=COLOR_Y)

	ax.tick_params(axis="x",labelsize=13)       	
	ax.tick_params(axis="y",labelsize=13)
	
	ax.legend(loc='lower center',fontsize=12, bbox_to_anchor=(0.5, 1.03),
          fancybox=True, shadow=False, ncol=5)
	ax.set_xlabel(r"$\beta^*$[cm]",fontsize = FONTSIZE)
	ax.set_ylabel(r"$\Delta\beta^{IP1}[\%]$",fontsize = FONTSIZE)


	for optics in opticsfiles_correction:
		beta_star_correction = _get_beta_star(proton, optics)*100
		ax.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color="black")
	
		
	plt.tight_layout()
	plt.savefig(f"plots/{savename}")
	plt.show()
	
	
def plot_lumi_optics_change(outputfile_dir, pickle_name, savename, proton, opticsfiles_correction = []):
	with open(f"{outputfile_dir}{pickle_name}","rb") as p:
		seed_dict = pickle.load(p)
	
	beta_star_l = []
	betx_IP1_l = []
	bety_IP1_l = []
	betx_IP5_l = []
	bety_IP5_l = []
	
	for i , key in enumerate(seed_dict):
		twiss_dict = seed_dict[key]
		
		
		for j , key in enumerate(twiss_dict):
			model_df = twiss_dict[key]["model"]
			
			df = twiss_dict[key]["meas"]
			
			if i == 0:
				betx_IP1_l.append([df.loc["IP1","BETX"]])
				bety_IP1_l.append([df.loc["IP1","BETY"]])
				betx_IP5_l.append([df.loc["IP5","BETX"]])
				bety_IP5_l.append([df.loc["IP5","BETY"]])
				beta_star_l.append(_get_beta_star(proton, key))
			else:
				betx_IP1_l[j].append(df.loc["IP1","BETX"])
				bety_IP1_l[j].append(df.loc["IP1","BETY"])
				betx_IP5_l[j].append(df.loc["IP5","BETX"])
				bety_IP5_l[j].append(df.loc["IP5","BETY"])
	
	#converting to numpy array
	beta_star_l = np.array(beta_star_l) * 100
	betx_IP1_l = np.array(betx_IP1_l)
	bety_IP1_l = np.array(bety_IP1_l)
	betx_IP5_l = np.array(betx_IP5_l)
	bety_IP5_l = np.array(bety_IP5_l)
	
	
	IP1_lumi = 1 / np.sqrt(betx_IP1_l * bety_IP1_l)
	IP5_lumi = 1 / np.sqrt(betx_IP5_l * bety_IP5_l)
	lumi_imbalance_l = 100 * (IP1_lumi - IP5_lumi) / IP5_lumi
	
	#average
	lumi_imbalance_avg_l = np.mean(lumi_imbalance_l,axis=1)
	
	#standard deviation
	lumi_imbalance_std_l = np.std(lumi_imbalance_l,axis=1)
	
	#max
	
	lumi_imbalance_max_l = absmaxND(lumi_imbalance_l,axis=1)
		
	fig , ax = plt.subplots()
	ax.errorbar(beta_star_l, lumi_imbalance_avg_l,yerr=lumi_imbalance_std_l, marker = "+", label = r"$\frac{L_{IP1}}{L_{IP5}}$",color=COLOR_X)
	ax.plot(beta_star_l, lumi_imbalance_max_l,"x", label = r"max $\frac{L_{IP1}}{L_{IP5}}$",color=COLOR_X)
	ax.legend(fontsize = 20)
	ax.set_xlabel(r"$\beta^*$[cm]",fontsize = FONTSIZE)
	ax.set_ylabel(r"Luminosity [%]",fontsize = FONTSIZE)
	ax.tick_params(axis="x",labelsize=16)       	
	ax.tick_params(axis="y",labelsize=16)
	
	for optics in opticsfiles_correction:
		beta_star_correction = _get_beta_star(proton, optics)*100
		ax.axvline(x=beta_star_correction,linestyle="--",zorder=1,alpha=0.5,color="black")
	
	plt.tight_layout()
	plt.savefig(f"plots/{savename}")
	plt.show()	
	
	
#utilities


def absmaxND(a, axis=None):
    amax = a.max(axis)
    amin = a.min(axis)
    return np.where(-amin > amax, amin, amax)

def _get_IP_beta_beat(df, model_df, plane, IP):
	bet_model = model_df.loc[IP,f"BET{plane}"]
	bet = df.loc[IP,f"BET{plane}"]
	return 100 * (bet - bet_model) / bet_model

def _get_beta_beat(df,model_df,plane,method = "rms"):
	method_dict = {
	"rms" : lambda x : np.sqrt(np.mean(x**2)),
	"max" : lambda x : max(abs(x))
		}
		
	
	bet_model = model_df[f"BET{plane}"].to_numpy()
	bet = df[f"BET{plane}"].to_numpy()
	bb = 100 * (bet - bet_model) / bet_model
	
	return method_dict[method](bb)
		
def _get_beta_star( proton, opticsfile):
	file1 = open(f"{proton}{opticsfile}")
	beta_star = 0
	for line in file1.readlines():
		if line.startswith("betx_IP1"):
			beta_star = line.split(":=")[1].split(";")[0]
			break
	return float(beta_star)
	
