from pathlib import Path

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

"""following paths must be changed"""
LHC_PATH = "/afs/cern.ch/eng/lhc/optics/runII/2018" #path to lhc folder
PROTON = f"/afs/cern.ch/user/s/sfar/public/forTobias/" #folder where opticsfiles are stored
SEQEDIT = "/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/IR7-Run3seqedit.madx" #edits sequence from run2 to run3

WORKING_DIR = str((Path(__file__).parent).absolute()) + "/"
OUTPUTFILE_DIR = f"{WORKING_DIR}outputfiles/"
BASE_SCRIPT_DIR = f"{WORKING_DIR}base_scripts/"
BASE_SCRIPT = f"{BASE_SCRIPT_DIR}REALISTIC_ERROR.madx"
OPTICSFILE = "opticsfile.180"
QX = 0.31 
QY = 0.32
QX_DRIVEN = 0.30
QY_DRIVEN = 0.332

MODEL_DIR = f"{WORKING_DIR}model_{OPTICSFILE}_Qx{QX}_Qy{QY}/"
RESPONSEMATRIX = f"{OUTPUTFILE_DIR}fullresponse{OPTICSFILE}_Qx{QX}_Qy{QY}"
ACCEL_SETTINGS = dict(ats=True,beam=1, year="2018", accel="lhc", energy=6.5)

VARIABLE_CATEGORIES=["MQM","MQT","MQTL","MQY"]

"""
creating model and responsematrix. 
PS : Must be called manually for all optics where you want to correct, change OPTICSFILE and run create_model_and_response, note that the name of MODEL_DIR and RESPONSEMATRIX depends on OPTICSFILE
"""
OPTICSFILE = "opticsfile.120"
MODEL_DIR = f"{WORKING_DIR}model_{OPTICSFILE}_Qx{QX}_Qy{QY}/"
RESPONSEMATRIX = f"{OUTPUTFILE_DIR}fullresponse{OPTICSFILE}_Qx{QX}_Qy{QY}"
create_model_and_response(WORKING_DIR, OUTPUTFILE_DIR, MODEL_DIR, RESPONSEMATRIX, OPTICSFILE, ACCEL_SETTINGS, QX, QY, QX_DRIVEN, QY_DRIVEN,VARIABLE_CATEGORIES, PROTON, LHC_PATH, SEQEDIT, append_sequedit=True)


OPTICSFILE = "opticsfile.150"
MODEL_DIR = f"{WORKING_DIR}model_{OPTICSFILE}_Qx{QX}_Qy{QY}/"
RESPONSEMATRIX = f"{OUTPUTFILE_DIR}fullresponse{OPTICSFILE}_Qx{QX}_Qy{QY}"
create_model_and_response(WORKING_DIR, OUTPUTFILE_DIR, MODEL_DIR, RESPONSEMATRIX, OPTICSFILE, ACCEL_SETTINGS, QX, QY, QX_DRIVEN, QY_DRIVEN,VARIABLE_CATEGORIES, PROTON, LHC_PATH, SEQEDIT, append_sequedit=True)

OPTICSFILE = "opticsfile.180"
MODEL_DIR = f"{WORKING_DIR}model_{OPTICSFILE}_Qx{QX}_Qy{QY}/"
RESPONSEMATRIX = f"{OUTPUTFILE_DIR}fullresponse{OPTICSFILE}_Qx{QX}_Qy{QY}"
create_model_and_response(WORKING_DIR, OUTPUTFILE_DIR, MODEL_DIR, RESPONSEMATRIX, OPTICSFILE, ACCEL_SETTINGS, QX, QY, QX_DRIVEN, QY_DRIVEN,VARIABLE_CATEGORIES, PROTON, LHC_PATH, SEQEDIT, append_sequedit=True)


CORRECTION_DICT_PICKLE = "correction_dict.pickle"
OPTICSFILES = [f"opticsfile.{i}" for i in range(110,191,5)]
OPTICSFILES_CORRECTION = ["opticsfile.120","opticsfile.150","opticsfile.180"]
SEEDS = [f"{i}" for i in range(1)]
CHANGE_DICT2_WITH_FEEDOWN = {
	"%seqedit" : SEQEDIT,
	"%feeddown" : "1",
	"%correct" : "0",
	"%focusing_error" : "1",
	"%SEED" : "6",
	"%PROTON" : PROTON, #optic2022
        "%lhc_path" : LHC_PATH,
        "%opticsfile" :   OPTICSFILE,
        "%working_dir" : WORKING_DIR,
        "%twiss_pattern" :  "BPM",
        "%QX" :  str(62 + float(QX)),
        "%QY" :  str(60 + float(QY))
}

"""plot betabeating before and after correction"""
ADD_NOISE = True
PLANE = "X"
PICKLE_NAME = "corrections.pickle"
#get_corrections(BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, MODEL_DIR, RESPONSEMATRIX, OUTPUTFILE_DIR, ACCEL_SETTINGS, PICKLE_NAME, VARIABLE_CATEGORIES, add_noise=ADD_NOISE, iterations=2, optics_params=["PHASEX","PHASEY","BETX","BETY","Q"], weights=[1.,1.,0.,0.,0.])
#plot_beta_beat(OUTPUTFILE_DIR, PICKLE_NAME, MODEL_DIR, f"betabeat_noise{ADD_NOISE}_plane{PLANE}_withfeeddown_noiseBPM.pdf", plane=PLANE)


"""
get_correction_dict: makes a pickle of corrections at the opticsfiles specified in OPTICSFILES_CORRECTION for each seed given in seeds
"""
get_correction_dict(CORRECTION_DICT_PICKLE, SEEDS, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, ACCEL_SETTINGS, VARIABLE_CATEGORIES, QX, QY, add_noise = ADD_NOISE, iterations=2, optics_params=["PHASEX","PHASEY","BETX","BETY","Q"], weights=[1.,1.,5.,5.,10.])

CORRECTION_PATH = f"{WORKING_DIR}changeparameters.madx" #correction from real measurement

"""plot correlation of corrections at the same seed for diferent optics"""
plot_correction_correlation(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, ["MQM","MQY"], PROTON, opticsfile_comparison="opticsfile.180")
plot_correction_correlation(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, ["MQM","MQY"], PROTON, opticsfile_comparison="opticsfile.150")

""""Plot histogram of max and rms magnet strength for each seed"""
plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.120", VARIABLE_CATEGORIES, CORRECTION_PATH,  method = "max")
plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.150", VARIABLE_CATEGORIES,CORRECTION_PATH, method = "max")
plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.180", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "max")

plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.120", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")
plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.150", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")
plot_magnet_strength(CORRECTION_DICT_PICKLE, OUTPUTFILE_DIR, "opticsfile.180", VARIABLE_CATEGORIES,CORRECTION_PATH,  method = "rms")

"""
get_twiss_optics evaluates the correction for the optics given in opticsfiles. Corrections taken from get_correction_dict and linearly weights the correction based on the beta*. Pickles dictionary of the twiss dataframes after correction. 

then plot rms, max betabeating, betabeat at IP1 and IP5, betabeating at IP8 and the luminosity imbalance between IP1 and IP5 against the beta* at IP1 and IP5. Mean and standard deviation accros the seeds is plotted aswell as max value accros the seeds.  
"""
PICKLE_NAME = "optics_change1_withFeedownIP8.pickle"
get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", PROTON, method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_604530.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)


PICKLE_NAME = "optics_change2_withFeedownIP8.pickle"
OPTICSFILES_CORRECTION = ["opticsfile.120","opticsfile.180"]
get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", PROTON, method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_6030.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)



PICKLE_NAME = "optics_change4_withFeedownIP8.pickle"#"optics_change4_withFeedownIP8_test.pickle"
OPTICSFILES_CORRECTION = ["opticsfile.150"]
get_twiss_optics(CORRECTION_DICT_PICKLE, OPTICSFILES, OPTICSFILES_CORRECTION, BASE_SCRIPT, CHANGE_DICT2_WITH_FEEDOWN, OUTPUTFILE_DIR, PICKLE_NAME)
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"rms_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION,method="rms")
plot_optics_change(OUTPUTFILE_DIR, PICKLE_NAME,  f"max_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", PROTON, method="max", opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_IP8_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"IP8_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)
plot_lumi_optics_change(OUTPUTFILE_DIR, PICKLE_NAME, f"lumi_betabeat_noise{ADD_NOISE}_withFeedown_45.pdf", PROTON, opticsfiles_correction = OPTICSFILES_CORRECTION)


"""
Plots ilustration of the weighting of the corrections
"""
OPTICSFILES = [f"opticsfile.{i}" for i in range(109,192,1)]
OPTICSFILES_CORRECTION = ["opticsfile.121","opticsfile.151","opticsfile.181"]
plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_304560.pdf", PROTON)
OPTICSFILES_CORRECTION = ["opticsfile.121","opticsfile.181"]
plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_3060.pdf", PROTON)
OPTICSFILES_CORRECTION = ["opticsfile.151"]
plot_correction_ramp(OPTICSFILES, OPTICSFILES_CORRECTION, "weight_45.pdf", PROTON)
