from beta_beat_estimation import  create_model_and_response
from beta_beat_estimation import  get_corrections
from beta_beat_estimation import get_twiss_optics
from beta_beat_estimation import get_correction_dict

from plot_beta_beat_estimation import plot_beta_beat
from plot_beta_beat_estimation import plot_optics_change
from plot_beta_beat_estimation import plot_IP_optics_change
from plot_beta_beat_estimation import plot_lumi_optics_change
from plot_beta_beat_estimation import plot_magnet_strength
from plot_beta_beat_estimation import plot_correction_ramp
from plot_beta_beat_estimation import plot_correction_correlation
from plot_beta_beat_estimation import plot_IP8_optics_change

LHC_PATH = f"/home/eirik/CERN/lhc2018/2018/"
OUTPUTFILE_DIR = "outputfiles/"
BASE_SCRIPT_DIR = "/home/eirik/CERN/beta_beat_estimation/base_scripts/"
BASE_SCRIPT = f"{BASE_SCRIPT_DIR}REALISTIC_ERROR.madx"
OPTICSFILE = "opticsfile.180"
QX = 0.31 
QY = 0.32
QX_DRIVEN = 0.30
QY_DRIVEN = 0.332
WORKING_DIR = "/home/eirik/CERN/beta_beat_estimation/"
MODEL_DIR = f"model_{OPTICSFILE}_Qx{QX}_Qy{QY}/"
RESPONSEMATRIX = f"{OUTPUTFILE_DIR}fullresponse{OPTICSFILE}_Qx{QX}_Qy{QY}"
PROTON = "optic2022"
ACCEL_SETTINGS = dict(ats=True,beam=1, year="2018", accel="lhc", energy=6.5)

VARIABLE_CATEGORIES=["MQM","MQT","MQTL","MQY"]

"""creating model and responsematrix. Must be called manually for all optics where you want to correct"""
#create_model_and_response(WORKING_DIR, OUTPUTFILE_DIR, MODEL_DIR, RESPONSEMATRIX, OPTICSFILE, ACCEL_SETTINGS, QX, QY, QX_DRIVEN, QY_DRIVEN,VARIABLE_CATEGORIES,PROTON = PROTON,append_sequedit=True)


CHANGE_DICT2 = {
	"%feeddown" : "1",
	"%correct" : "0",
	"%focusing_error" : "1",
	"%SEED" : "6",
	"%PROTON" : PROTON, #optic2022
        "%lhc_path" : LHC_PATH,
        "%opticsfile" :   OPTICSFILE,
        "%twiss_pattern" :  "BPM",
        "%QX" :  str(62 + float(QX)),
        "%QY" :  str(60 + float(QY))
}


"""plot betabeating before and after correction"""
ADD_NOISE = True
PLANE = "X"
PICKLE_NAME = "correction1.pickle"
#get_corrections(BASE_SCRIPT, CHANGE_DICT2, MODEL_DIR, RESPONSEMATRIX, OUTPUTFILE_DIR, ACCEL_SETTINGS, PICKLE_NAME, VARIABLE_CATEGORIES, add_noise=ADD_NOISE, iterations=2, optics_params=["MUX","MUY","BETX","BETY","Q"], weights=[1.,1.,5.,5.,10.])
#plot_beta_beat(OUTPUTFILE_DIR, PICKLE_NAME, MODEL_DIR, f"betabeat_noise{ADD_NOISE}_plane{PLANE}_withfeeddown_noiseBPM.pdf", plane=PLANE)





CORRECTION_DICT_PICKLE = "correction_dict_withFeedown.pickle"
OPTICSFILES = [f"opticsfile.{i}" for i in range(110,191,5)]
OPTICSFILES_CORRECTION = ["opticsfile.120","opticsfile.150","opticsfile.180"]
SEEDS = [f"{i}" for i in range(50)]#["987456321","897456321","111111111","222222222","333333333","444444444","555555555","666666666","777777777","888888888","999999999"]
CHANGE_DICT2_WITH_FEEDOWN = {
	"%feeddown" : "1",
	"%correct" : "0",
	"%focusing_error" : "1",
	"%SEED" : "6",
	"%PROTON" : PROTON, #optic2022
        "%lhc_path" : LHC_PATH,
        "%opticsfile" :   OPTICSFILE,
        "%twiss_pattern" :  "BPM",
        "%QX" :  str(62 + float(QX)),
        "%QY" :  str(60 + float(QY))
}

"""
get_correction_dict: makes a pickle of corrections at the opticsfiles specified in OPTICSFILES_CORRECTION for each seed given in seeds
"""
#get_correction_dict(CORRECTION_DICT_PICKLE, SEEDS, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, ACCEL_SETTINGS, VARIABLE_CATEGORIES, QX, QY, add_noise = ADD_NOISE, iterations=2, optics_params=["MUX","MUY","BETX","BETY","Q"], weights=[1.,1.,5.,5.,10.])

CORRECTION_PATH = "/home/eirik/CERN/beta_beat_estimation/changeparameters.madx"

"""plot correlation of corrections at the same seed for diferent optics"""
#plot_correction_correlation(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, ["MQM","MQY"],opticsfile_comparison="opticsfile.180")
#plot_correction_correlation(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, ["MQM","MQY"],opticsfile_comparison="opticsfile.150")

""""Plot histogram of max and rms magnet strength for each seed"""
#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.120", VARIABLE_CATEGORIES, CORRECTION_PATH,  method = "max")
#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.150", VARIABLE_CATEGORIES,CORRECTION_PATH, method = "max")
#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.180", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "max")

#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.120", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")
#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.150", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")
#plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.180", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")

"""
get_twiss_optics evaluates the correction for the optics given in opticsfiles. Corrections taken from get_correction_dict and linearly weights the correction based on the beta*. Pickles dictionary of the twiss dataframes after correction. 

then plot rms, max betabeating, betabeat at IP1 and IP5, betabeating at IP8 and the luminosity imbalance between IP1 and IP5 against the beta* at IP1 and IP5. Mean and standard deviation accros the seeds is plotted aswell as max value accros the seeds.  
"""
PICKLE_NAME = "optics_change1_withFeedownIP8.pickle"
#get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf",method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)


PICKLE_NAME = "optics_change2_withFeedownIP8.pickle"
OPTICSFILES_CORRECTION = ["opticsfile.120","opticsfile.180"]
#get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf",method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)



PICKLE_NAME = "optics_change4_withFeedownIP8.pickle"
OPTICSFILES_CORRECTION = ["opticsfile.150"]
#get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
#plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf",method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)
#plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", opticsfiles_correction = OPTICSFILES_CORRECTION)


"""

"""
OPTICSFILES = [f"opticsfile.{i}" for i in range(109,192,1)]
OPTICSFILES_CORRECTION = ["opticsfile.121","opticsfile.151","opticsfile.181"]
#plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_304560.pdf")
OPTICSFILES_CORRECTION = ["opticsfile.121","opticsfile.181"]
#plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_3060.pdf")
OPTICSFILES_CORRECTION = ["opticsfile.151"]
#plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_45.pdf")
