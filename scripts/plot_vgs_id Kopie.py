import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from multiprocessing import Pool
from multiprocessing import Value
import matplot2tikz


import numpy as np
import pandas as pd
from pathlib import Path

#Naming after Parameters (all Measurments with Neg and Pos Pol)
B = "B2"
S = "S3"
K = "K2" 
D = "O"
PA = ["pol4", "polneg3"]
TA = ["ptype", "ntype"]
# unit for ID
UA = (1e6, "$\mu$")

#create folders

Path(f"Results/{B}-{S}-{K}-{D}").mkdir(exist_ok=True)
try:
    Path(f"Results/{B}-{S}-{K}-{D}/vgsid").mkdir()
except FileExistsError:
    ans = input("Directory existing overwrite Images? (y/n)")
    if ans.lower() == "y":
        pass
    else:
        raise SystemExit("No Overwrite, EXIT")

Path(f"Results/{B}-{S}-{K}-{D}/vgsid/pgf").mkdir(exist_ok=True)
Path(f"Results/{B}-{S}-{K}-{D}/vgsid/png").mkdir(exist_ok=True)
Path(f"Results/{B}-{S}-{K}-{D}/vgsid/grid").mkdir(exist_ok=True)
Path(f"Results/{B}-{S}-{K}-{D}/vgsid/comp").mkdir(exist_ok=True)


# Plot functions   
def logmaker(fig : Figure, ax: Axes, T):
    for i in  ax.get_lines():
        y= np.abs(i.get_ydata()/(UA[0]))
        
        y[y <= 1e-10] = 1e-10
        i.set_ydata(y)
        x= np.abs(i.get_xdata())
        x[x == 0] = 1e-12
        i.set_xdata(x)
    if T == "ntype":
        ax.set_xlim(0, 2)
    elif T == "ptype":
        ax.set_xlim(0, 1.5)
    ax.set_ylim(1e-10, 1e-3)  
    ax.set_yscale("log", base =10)
    ax.yaxis.set_major_locator(LogLocator(base=10))  
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs=np.arange(2,10)*0.1)) 
    ax.xaxis.set_major_locator(MultipleLocator(0.2))  
    ax.xaxis.set_minor_locator(MultipleLocator(0.05)) 
    ax.set_ylabel(fr"|$I_D$|[A]")
    ax.set_xlabel(fr"|$V_{{DS}}$|[V]")

    ax.grid(linestyle="--", linewidth=0.2, color='.25', which="minor", axis="y")
    ax.grid(linestyle="-", linewidth=0.5, color='.25', which="major", axis="y")
    ax.grid(False, which="both", axis="x")

    return fig, ax 

def pltvdsid(data, T, P=0, w=True):
    fig, ax = plt.subplots()
    ax.set_xlabel(fr"$V_{{DS}}$[V]")
    ax.set_ylabel(fr"$I_D$[{UA[1]}A]")


    ax.yaxis.set_major_locator(MultipleLocator(10))  
    ax.yaxis.set_minor_locator(MultipleLocator(5)) 

    ax.xaxis.set_major_locator(MultipleLocator(0.2))  
    ax.xaxis.set_minor_locator(MultipleLocator(0.05)) 
    if(w == True):
        table = data[(T,P)]
        i =  int(table.shape[1]/8)
        while(i != 0):
            ax.plot(table["DrainV(1)"], table[fr"DrainI({i})"]*UA[0], label=f"{table[f'GateV({i})'].iloc[0]} V", lw=0.5)
            i += -1
        ax.legend(loc="upper left", title=r"$V_{GS}$")
        if T == "ntype":
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 150)  
        elif T == "ptype":
            ax.set_xlim(-1.5, 0)
            ax.set_ylim(-180, 0)  
    return fig, ax  

def compare(data, T):
    fig, ax =pltvdsid(data, T, w=False)
    for i in PA :
        table = data[(T,i)]
        numb =  [1, 2, 3, 5, int(table.shape[1]/8)]
        if i == "pol4":
            polval =  4
            col = ["#3b0000", "#8c1d1d", "#d73027", "#f46d43", "#ff9a7a"]
        elif i == "polneg3":
            polval = -3
            col = ["#001a3a", "#0b3c7a", "#1f6fd2", "#4a90e2", "#7fb3ff"]
        c = 0
        for j in numb:
            ax.plot(table["DrainV(1)"], table[fr"DrainI({j})"]*UA[0], label=f"{table[f'GateV({j})'][1]} V, pol.: {polval} V ",c=f'{col[c]}', lw=0.5)
            c += 1
        if T == "ntype":
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 150)  
        elif T == "ptype":
            ax.set_xlim(-1.5, 0)
            ax.set_ylim(-180, 0)  
        ax.legend()
    return fig, ax


def multi(args):
    #VDS ID graphs
    T, P, data = args
    fig, ax = pltvdsid(data,T, P)
   # fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/pgf/vds-id-{P}-{T}-{B}-{K}-{S}-{D}.pgf")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/png/vds-id-{P}-{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', which="both")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/grid/grid-vds-id-{P}-{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    fig, ax = logmaker(fig, ax, T)
    #fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/pgf/log-vds-id-{P}-{T}-{B}-{K}-{S}-{D}.pgf")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/png/log-vds-id-{P}-{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    plt.close(fig)


def multic(a2):
    T, data = a2
    fig, ax = compare(data,T)
   # fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/comp/{T}-{B}-{K}-{S}-{D}.pgf")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/comp/{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    ax.grid(linestyle="--", linewidth=0.5, color='.25', which="both")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/comp/grid-{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    fig, ax = logmaker(fig, ax, T)
    #fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/comp/log-{T}-{B}-{K}-{S}-{D}.pgf")
    fig.savefig(f"Results/{B}-{S}-{K}-{D}/vgsid/comp/log-{T}-{B}-{K}-{S}-{D}.png", dpi=300)
    plt.close(fig)


#MAIN
#Reading data
count = 0
data = {}
for T in TA:
    for P in PA:
        table = pd.read_csv(f"data/{B}-{S}-{K}-{D}/vds-id-{P}-{T}-{B}-{K}-{S}-{D}.csv",sep=";" ,decimal="," )
        table = table.dropna(axis=1, how="all")
        data[(T,P)] = table

if __name__ == "__main__":
    args = [(T, P, data) for T in TA for P in PA]
    a2 = [(T, data) for T in TA]
    with Pool(4) as p:
        p.map(multi, args)
    with Pool(2) as p:
        p.map(multic, a2)









