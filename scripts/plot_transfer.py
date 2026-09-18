import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplot2tikz
from pathlib import Path
from scipy.signal import savgol_filter

B = "B2"
S = "S3"
K = "K2" 
D = "O"
PA = ["VD01","VD12"]
# unit for ID
UA = (1e6, "$\mu$")

#create folders
Path(f"Results/{B}-{S}-{K}-{D}").mkdir(exist_ok=True)
try:
    Path(f"Results/{B}-{S}-{K}-{D}/transfer").mkdir()
except FileExistsError:
    ans = input("Directory existing overwrite Images? (y/n)")
    if ans.lower() == "y":
        pass
    else:
        raise SystemExit("No Overwrite, EXIT")


#reading Data
c = ["r","g","b"]
fges, ages = plt.subplots(figsize=(10, 4))
cit = 0
for P in PA:
    table = pd.read_csv(f"data/{B}-{S}-{K}-{D}/transfer-{P}-{B}-{S}-{K}-{D}.csv",sep=";" ,decimal="," )
    table = table.dropna(axis=1, how="all")
    maxpos = table["GateV"].idxmax()
    t1 = table.loc[:maxpos]
    t2 = table.loc[maxpos+1:]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlabel(fr"$V_{{G}}$[V]")
    ax.set_ylabel(fr"$|I_D|$[A]")
    ax.plot(t1["GateV"], np.abs(t1["DrainI"]),"-",c="b")
    ax.plot(t2["GateV"], np.abs(t2["DrainI"]),"-",c="b")
    ages.plot(t1["GateV"], np.abs(t1["DrainI"]),"-",c=c[cit])
    ages.plot(t2["GateV"], np.abs(t2["DrainI"]),"-",c=c[cit])
    ax.set_ylim(1e-12, 1e-4)  
    ax.set_yscale("log", base =10)
    ax.yaxis.set_major_locator(LogLocator(base=10))  
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*0.1)) 
    ax.xaxis.set_major_locator(MultipleLocator(0.5))  
    ax.xaxis.set_minor_locator(MultipleLocator(0.25)) 
    ax.set_xlim(-3, 4.5)
    ax.grid(linestyle="--", linewidth=0.2, color='.25', which="minor", axis="y")
    ax.grid(linestyle="-", linewidth=0.5, color='.25', which="major", axis="y")
    
    step = 92
    x = t1["GateV"].values
    y = np.abs(t1["DrainI"].values)

    for i in range(0, len(x)-1, step):
        ax.annotate(
            "",
            xy=(x[i+1], y[i+1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.8,
                color="Blue",
            )
        )

    x = t2["GateV"].values
    y = np.abs(t2["DrainI"].values)

    for i in range(0, len(x)-1, step):
        ax.annotate(
            "",
            xy=(x[i+1], y[i+1]),
            xytext=(x[i], y[i]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.8,
                color="Blue",
            )
        )

    ax.grid(False, which="both", axis="x")
    fig.subplots_adjust(bottom=0.2)
    matplot2tikz.save(f"Results/{B}-{S}-{K}-{D}/transfer/{P}-{B}-{K}-{S}-{D}.tex", figure=fig)
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/transfer/{P}-{B}-{K}-{S}-{D}.png", dpi=300)
    cit+=1
ages.set_ylim(1e-12, 1e-4)  
ages.set_yscale("log", base =10)
ages.yaxis.set_major_locator(LogLocator(base=10))  
ages.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*0.1)) 
ages.xaxis.set_major_locator(MultipleLocator(0.5))  
ages.xaxis.set_minor_locator(MultipleLocator(0.25)) 
ages.set_xlim(-3, 4.5)
ages.set_xlabel(fr"$V_{{G}}$[V]")
ages.set_ylabel(fr"$|I_D|$[A]")
ages.grid(linestyle="--", linewidth=0.2, color='.25', which="minor", axis="y")
ages.grid(linestyle="-", linewidth=0.5, color='.25', which="major", axis="y")
fges.subplots_adjust(bottom=0.2)
matplot2tikz.save(f"Results/{B}-{S}-{K}-{D}/transfer/comp-{P}-{B}-{K}-{S}-{D}.tex", figure=fges)
fges.savefig(f"Results/{B}-{S}-{K}-{D}/transfer/comp-{P}-{B}-{K}-{S}-{D}.png", dpi=300)