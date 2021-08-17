from pathlib import Path

import numpy as np
import tfs
import copy
import pickle
import os


import omc3.madx_wrapper as madx_wrapper
from omc3.global_correction import global_correction_entrypoint
from omc3.correction.constants import ERR, DELTA
from omc3.response_creator import create_response_entrypoint
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import (BETA_NAME, DISPERSION_NAME,
                                                EXT, NORM_DISP_NAME,
                                                PHASE_NAME)
#from omc3.scripts.fake_measurement from model import generate as generate_fake


from optics_functions.coupling import coupling_via_cmatrix
from optics_functions.utils import prepare_twiss_dataframe

#constants
CORRECT_FILE = "changeparameters_iter_correct.madx"     
DELTA_Q_MIN = "deltaQmin"

NAME = "NAME"


def create_model_and_response(working_dir, outputfile_dir, model_dir, responsematrix, opticsfile, accel_settings, qx, qy, qx_driven, qy_driven, variable_categories, proton, lhc_path, seqedit, append_sequedit=False):
	#accel settings is bad code..
	if append_sequedit:
		modifiers_path = f"{outputfile_dir}modifiers.madx"
		modifiers_file = f"call,file=\"{seqedit}\";\n call,file=\"{proton}{opticsfile}\";"
		with open(modifiers_path,"w") as f:
			f.write(modifiers_file) 
	else:
		modifiers_path = f"{proton}{opticsfile}"
	
	
	create_instance_and_model(
	accel = accel_settings["accel"],
	year = accel_settings["year"],
	energy = accel_settings["energy"],
	beam = accel_settings["beam"],
	type="nominal",
	ats = True,
	nat_tunes = [qx, qy],
	drv_tunes = [qx_driven, qy_driven],
	driven_excitation ="acd",
	dpp = 0.,
	modifiers = [Path(modifiers_path).absolute()],
	outputdir = Path(model_dir).absolute(),
	)

	create_response_entrypoint(
		**accel_settings,
		model_dir = Path(model_dir).absolute(),
		delta_k = 0.00002,
		variable_categories = variable_categories,\
		outfile_path = Path(responsematrix).absolute(),
		)

def get_correction_dict(correction_dict_pickle, seeds, opticsfiles_correction, base_script, change_dict, outputfile_dir, accel_settings, variable_categories, QX, QY, add_noise = False, iterations=1, optics_params=["MUX","MUY"],weights=[1.,1.]):
	
	correction_dict_seed = {}
	iteration_dict = {opticsfile:0 for opticsfile in opticsfiles_correction}
	for seed in seeds:
		change_dict_seed = copy.deepcopy(change_dict)
		_change_value(change_dict_seed,"%SEED",seed)
		
		change_dict_local1 = copy.deepcopy(change_dict_seed)
		_change_value(change_dict_local1,"%twiss_pattern","BPM")
		
		correction_dict_optics = {}
		for opticsfile in opticsfiles_correction:
			_change_value(change_dict_local1,"%opticsfile",opticsfile)
			model_dir = f"model_{opticsfile}_Qx{QX}_Qy{QY}/"
			responsematrix = f"{outputfile_dir}fullresponse{opticsfile}_Qx{QX}_Qy{QY}"
			correction , iterated = _get_correction_string(base_script, change_dict_local1, model_dir, responsematrix, outputfile_dir, accel_settings, iterations, 
		variable_categories, optics_params, weights, add_noise)
			correction_dict_optics[opticsfile] = correction
			
			if iterated:
				iteration_dict[opticsfile] += 1	
		correction_dict_seed[seed] = correction_dict_optics  
	
	with open(f"{outputfile_dir}iteration_dict.txt","w") as f:
		for opticsfile , count in iteration_dict.items():
			f.write('%s:%s\n' % (opticsfile, count))	
		
	with open(f"{outputfile_dir}{correction_dict_pickle}","wb") as f:
                pickle.dump(correction_dict_seed,f)

def get_twiss_optics(correction_dict_pickle, opticsfiles, opticsfiles_correction, base_script, change_dict, outputfile_dir ,pickle_name):
	with open(f"{outputfile_dir}{correction_dict_pickle}","rb") as p:
                correction_dict_seed = pickle.load(p)
	
	seed_dict = {}
	for seed in correction_dict_seed:
		
		change_dict_seed = copy.deepcopy(change_dict)
		_change_value(change_dict_seed,"%SEED",seed)
		
		correction_list = []
		for opticsfile in opticsfiles_correction:
			correction_list.append(correction_dict_seed[seed][opticsfile])
				
		change_dict_local2 = copy.deepcopy(change_dict_seed)
		twiss_dict = {}
		for opticsfile in opticsfiles:
			optics_dict = {}
				
			_change_value(change_dict_local2,"%opticsfile",opticsfile)
			#create a model
			_change_value(change_dict_local2,"%feeddown","0")	
			_change_value(change_dict_local2,"%focusing_error","0")
			_change_value(change_dict_local2,"%correct","0")
			optics_dict["model"] = _get_twiss_df(base_script, change_dict_local2, outputfile_dir)
			
			#make weighted correction
			_write_weighted_correction(opticsfile, opticsfiles_correction, correction_list, outputfile_dir)
			
			#create a measurement
			_change_value(change_dict_local2,"%feeddown","1")	
			_change_value(change_dict_local2,"%focusing_error","1")
			_change_value(change_dict_local2,"%correct","1")
			optics_dict["meas"]= _get_twiss_df(base_script, change_dict_local2, outputfile_dir)
			
			twiss_dict[opticsfile] = optics_dict
		
		seed_dict[seed] = twiss_dict
		
	with open(f"{outputfile_dir}{pickle_name}","wb") as f:
                pickle.dump(seed_dict,f)


                	

def get_corrections(base_script, change_dict, model_dir, responsematrix, outputfile_dir, accel_settings,pickle_name, variable_categories, add_noise = False, iterations=1, 
        optics_params=["MUX","MUY"],weights=[1.,1.]):
        
        correction_dict = _get_correction_dict(base_script, change_dict, model_dir, responsematrix, outputfile_dir, accel_settings, iterations, 
        variable_categories, optics_params, weights, add_noise)
        
        with open(f"{outputfile_dir}{pickle_name}","wb") as f:
                pickle.dump(correction_dict,f)



#Utilities
def _write_weighted_correction(opticsfile, opticsfiles_correction, correction_list, outputfile_dir):
	optics_number = _optics_number(opticsfile)
	#checking if the opticsfile is first of last
	os.system(f"rm {outputfile_dir}{CORRECT_FILE}")
	
	if optics_number <= _optics_number(opticsfiles_correction[0]):
		correction = correction_list[0]
		with open(f"{outputfile_dir}{CORRECT_FILE}","w") as f:	
			f.write(correction)
		return
	elif optics_number >= _optics_number(opticsfiles_correction[len(opticsfiles_correction)-1]):
		correction = correction_list[len(opticsfiles_correction)-1]
		with open(f"{outputfile_dir}{CORRECT_FILE}","w") as f:	
			f.write(correction)
		return
	
	#if opticsfile is in list or inbetween
	for i in range(1,len(opticsfiles_correction)):
		optics_number_correction1 = _optics_number(opticsfiles_correction[i - 1])
		optics_number_correction2 = _optics_number(opticsfiles_correction[i])
		if optics_number_correction1 == optics_number:
			correction_index = opticsfiles_correction.index(f"opticsfile.{optics_number}")
			correction = correction_list[correction_index]
			with open(f"{outputfile_dir}{CORRECT_FILE}","w") as f:	
				f.write(correction)
			return
		elif optics_number < optics_number_correction2:
			correction1 = correction_list[i-1]
			correction2 = correction_list[i]
			weight2 = (optics_number - optics_number_correction1) / (optics_number_correction2 - optics_number_correction1)
			weighted_correction = _get_weighted_correction(weight2,correction1,correction2)
			with open(f"{outputfile_dir}{CORRECT_FILE}","w") as f:	
				f.write(weighted_correction)
			return 

def _get_weighted_correction(weight2,correction1,correction2):
	weight1 = 1 - weight2
	weighted_correction1 = _scale_correction(weight1,correction1)
	weighted_correction2 = _scale_correction(weight2,correction2)
	
	weighted_correction = weighted_correction1 + weighted_correction2
	return weighted_correction
	
	
def _scale_correction(weight,correction):
	correction = correction.replace(" +", f" +({weight} * ")
	correction = correction.replace(" -", f" -({weight} * ")
	correction = correction.replace(";", f");")
	return correction		
			
def _optics_number(opticsfile):
	return int(opticsfile.split(".")[1])
	

def _get_correction_string(base_script, change_dict, model_dir, responsematrix, outputfile_dir, accel_settings, iterations, variable_categories, optics_params, weights, add_noise):
		
	change_dict_local = copy.deepcopy(change_dict)
	if "opticsfile.180" in model_dir:
		_change_value(change_dict_local,"%feeddown","0")

	correction_dict = _get_correction_dict(base_script, change_dict_local,model_dir, responsematrix, outputfile_dir, accel_settings, iterations, 
	variable_categories, optics_params, weights, add_noise)


	if len(correction_dict.keys()) == (iterations + 1):
		iterated = True
	else:
		iterated = False

	#add sextupole and do extra correction for 30cm
	if "opticsfile.180" in model_dir:

		_change_value(change_dict_local,"%feeddown","1")
		_change_value(change_dict_local,"%correct","1")

		with open(f"{outputfile_dir}{CORRECT_FILE}","r") as f:
			old_correction = f.read()
		
		model_df = tfs.read(f"{model_dir}twiss_elements.dat",index="NAME")
		twiss_df = _get_twiss_df(base_script, change_dict_local, outputfile_dir)
		if add_noise:
			_add_noise(twiss_df, model_df)  
		twiss_df  = twiss_df.drop("IP1")
		twiss_df  = twiss_df.drop("IP5")
		twiss_df  = twiss_df.drop("IP8")                   
		_tfs_converter(model_df, twiss_df, optics_params, outputfile_dir)	
		global_correction_entrypoint(**accel_settings,
				model_dir = Path(model_dir).absolute(),
				meas_dir=Path(outputfile_dir).absolute(),
				variable_categories=variable_categories,
				fullresponse_path=Path(responsematrix).absolute(),
				optics_params=optics_params,
				output_dir=Path(outputfile_dir).absolute(),
				weights=weights,
				svd_cut=0.01,
				max_iter=0)
                                                 	
		with open(f"{outputfile_dir}{CORRECT_FILE}","a") as f:
			f.write(old_correction)
		

	with open(f"{outputfile_dir}{CORRECT_FILE}") as f:
		correction = f.read()
	return correction , iterated


def _get_correction_dict(base_script, change_dict,model_dir, responsematrix, outputfile_dir, accel_settings, iterations, 
        variable_categories, optics_params, weights, add_noise, delete_changeparameter=True):
        
        #remove old corrections
        os.system(f"rm {outputfile_dir}changeparameters*")

        change_dict_local = copy.deepcopy(change_dict)
        model_df = tfs.read(f"{model_dir}twiss_elements.dat",index="NAME")
        twiss_df = _get_twiss_df(base_script, change_dict_local, outputfile_dir)
        
        correction_dict = {"measurement" : copy.deepcopy(twiss_df)}
        if add_noise:
        	_add_noise(twiss_df, model_df)   
        	   
        twiss_df  = twiss_df.drop("IP1")
        twiss_df  = twiss_df.drop("IP5")  
        twiss_df  = twiss_df.drop("IP8")                 
        _tfs_converter(model_df, twiss_df, optics_params, outputfile_dir)
        #correction
        iteration = 0
        correcting = True
        while correcting:
                #save previous correction
                if iteration > 0:
                        with open(f"{outputfile_dir}{CORRECT_FILE}","r") as f:
                                old_correction = f.read()
             
                global_correction_entrypoint(**accel_settings,
                                                 model_dir = Path(model_dir).absolute(),
                                                 meas_dir= Path(outputfile_dir).absolute(),
                                                 variable_categories = variable_categories,
                                                 fullresponse_path = Path(responsematrix).absolute(),
                                                 optics_params = optics_params,
                                                 output_dir = Path(outputfile_dir).absolute(),
                                                 weights = weights,
                                                 svd_cut = 0.01,
                                                 max_iter = 0)
                #append previous correction
                if iteration > 0:
                        with open(f"{outputfile_dir}{CORRECT_FILE}","a") as f:
                                f.write(old_correction)

                _change_value(change_dict_local,"%correct","1")
                corrected_df = _get_twiss_df(base_script, change_dict_local, outputfile_dir)
                correction_dict[f"{iteration}"] = copy.deepcopy(corrected_df)
                if _beta_tolerance(model_df, corrected_df) or iteration >= (iterations - 1):
                	correcting = False
                corrected_df  = corrected_df.drop("IP1")
                corrected_df  = corrected_df.drop("IP5")
                corrected_df  = corrected_df.drop("IP8")
                if add_noise:
                	_add_noise(corrected_df, model_df)
                _tfs_converter(model_df, corrected_df, optics_params, outputfile_dir)
                
               
                
                iteration += 1
        return correction_dict
        
def _beta_tolerance(model_df, corrected_df, max_tol = 0.15, IP_tol = 0.02):
	beating_df = (corrected_df.loc[:,["BETX","BETY"]] - model_df.loc[corrected_df.index,["BETX","BETY"]]) / model_df.loc[corrected_df.index,["BETX","BETY"]]        
	
	meeting_tol = True
	
	if abs(beating_df.loc["IP1","BETX"]) > IP_tol or  abs(beating_df.loc["IP1","BETY"]) > IP_tol:
		meeting_tol = False
	elif abs(beating_df.loc["IP5","BETX"]) > IP_tol or  abs(beating_df.loc["IP5","BETY"]) > IP_tol:
		meeting_tol = False
	elif max(abs(beating_df.loc[:,"BETX"].to_numpy())) > max_tol or  max(abs(beating_df.loc[:,"BETY"].to_numpy())) > max_tol:
		meeting_tol = False
	return meeting_tol
		
def _add_noise(df, model_df, amp = 1e-3):
	df["MUX"] +=  amp * np.random.normal(size = len(df["MUX"])) / np.sqrt(model_df.loc[list(df.index),"BETX"].to_numpy())
	df["MUY"] +=  amp * np.random.normal(size = len(df["MUX"])) / np.sqrt(model_df.loc[list(df.index),"BETY"].to_numpy())
	df["BETX"] += 1.5 * np.random.normal(size = len(df["BETX"]))
	df["BETY"] += 1.5 * np.random.normal(size = len(df["BETY"]))

def _change_value(change_dict, key, value):
	assert key in change_dict.keys(), f"{key} not in change_dict"
	change_dict[key] = value


def _get_twiss_df(base_script, change_dict, outputfile_dir, assert_list=[]):
        updated_script = _get_updated_script(base_script, change_dict, assert_list=assert_list)
        madx_wrapper.run_string(updated_script,log_file=f"{outputfile_dir}madx_log.txt")
        df = tfs.read(f"{outputfile_dir}twiss.dat",index="NAME")
        return df


def _get_updated_script(base_script, change_dict, assert_list=[]):
	with open(base_script,"r") as mask:	
		script = mask.read()
	
	for quantity in assert_list:
		assert quantity in script, f"{quantity} not in {base_script}"
	
	for key in change_dict:
		script = script.replace(key,change_dict[key])
	assert "%" not in script, f"% in script {base_script}"
	return script
	
	
def _tfs_converter(model_df, twiss_df, optics_parameters, output_dir):
    """
    Takes a twiss file and writes the parameters in optics_parameters to output_dir in the format global_correction_entrypoint uses (same format you would get from hole_in_one)
    Args:
        twiss_model_file:
        twiss_file:
        optics_parameters:
        output_dir:
    Returns:
    """
    err_low = 0.01
    err_high = 0.02

    h_dict = {"Q1": twiss_df["Q1"], "Q2": twiss_df["Q2"]}
    
    kmod_BPMs = ["BPMSW.1L5.B1","BPMSW.1R5.B1","BPMSW.1L1.B1","BPMSW.1R1.B1"]
    for parameter in optics_parameters:
        col = parameter if "MU" not in parameter else f"PHASE{parameter[-1]}"
        if parameter.startswith("PHASE"):
            new = tfs.TfsDataFrame(index=twiss_df.index[:-1:])  # ???????
            new[NAME] = twiss_df.index[:-1:]
            new[f"{NAME}2"] = twiss_df.index[1::]
            new[col] = (twiss_df.loc[new[f"{NAME}2"], f"MU{parameter[-1]}"].to_numpy() - twiss_df.loc[new[NAME], f"MU{parameter[-1]}"].to_numpy())
		
            new[f"{ERR}{col}"] = np.zeros(len(twiss_df.index) - 1)
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"]
            new[f"{DELTA}{col}"] = new[col] - (model_df.loc[new[f"{NAME}2"], f"MU{parameter[-1]}"].to_numpy() - model_df.loc[new[NAME], f"MU{parameter[-1]}"].to_numpy())
            write_file = f"{PHASE_NAME}{parameter[-1].lower()}{EXT}"

        elif parameter.startswith("BET"):
        
            new = tfs.TfsDataFrame(index=kmod_BPMs)
            new[NAME] = kmod_BPMs
            new[col] = twiss_df.loc[kmod_BPMs, parameter]

            new[f"{ERR}{col}"] = np.zeros(len(new.index))
            new[f"{ERR}{DELTA}{col}"] = new[f"{ERR}{col}"] / model_df.loc[kmod_BPMs, parameter]
            new[f"{DELTA}{col}"] = (new[col] - model_df.loc[kmod_BPMs, parameter]) / model_df.loc[kmod_BPMs, parameter]
            write_file = f"{BETA_NAME}{parameter[-1].lower()}{EXT}"
            
        tfs.write(f"{output_dir}{write_file}", new, headers_dict=h_dict, save_index="index_column")
        
        
        
