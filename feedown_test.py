import numpy as np
import tfs
import matplotlib.pyplot as plt


MODEL_DIR = MODEL_DIR = f"model_opticsfile.180_Qx0.31_Qy0.32/"

model_df = tfs.read(f"{MODEL_DIR}twiss.dat",index = "NAME")

meas_df = tfs.read(f"outputfiles/twiss_feedown.dat",index = "NAME")


#plt.plot(meas_df["S"], meas_df["X"],label="orbit X")
#plt.plot(meas_df["S"], meas_df["Y"],label="orbit Y")
##plt.legend()
##plt.savefig("plots/orbit.pdf")
#plt.show()

bbx = 100 * (meas_df.loc[model_df.index,"BETX"] - model_df.loc[:,"BETX"]) / model_df.loc[:,"BETX"]
bby = 100 * (meas_df.loc[model_df.index,"BETY"] - model_df.loc[:,"BETY"]) / model_df.loc[:,"BETY"]

plt.plot(model_df.loc[:,"S"],bbx,label="BETX")
plt.plot(model_df.loc[:,"S"],bbx,label="BETY")
plt.xlabel("position [m]")
plt.ylabel(r"$\Delta \beta \,$[%]")
plt.legend()
plt.savefig("plots/betabeat_feedown.pdf")
plt.show()
