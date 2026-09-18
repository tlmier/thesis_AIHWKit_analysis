import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplot2tikz
from scipy.spatial import cKDTree


file = "SG_2"

beg = 0.00
end = 1

vread_th = 2.4
vread_thp = 2.2
vd = 0.7

min_dt = 185e-6
max_dt = 215e-6

ima = 2.5e-7
imi = 0e-7

t0 = 300e-6


out_dir = Path(f"Results/GM/{file}")
out_dir.mkdir(parents=True, exist_ok=True)

table = pd.read_csv(f"data/Courve/{file}.csv", sep=",", decimal=".")

time = table["time"].to_numpy()
v1 = table["v1"].to_numpy()
i4 = table["i4"].to_numpy()


fig, ax = plt.subplots()

ax.plot(
    time,
    i4,
    "-",
    c="darkblue",
    linewidth=0.5
    )

ax.set_xlabel("time. [s]")
ax.set_ylabel("I_d [A]")
ax.set_ylim(-0.25e-6, 0.25e-6)  
fig.savefig(out_dir / f"{file}decay.png", dpi=900)
matplot2tikz.save(out_dir / f"{file}_decay.tex", figure=fig)
plt.close(fig)

base = (
    (time >= beg) &
    (time <= end) &
    (v1 <= vread_th) &
    (v1 >= vread_thp)
)


blocks = []
sel = np.zeros(len(table), dtype=bool)

start = None

for k, ok in enumerate(base):
    if ok and start is None:
        start = k

    elif not ok and start is not None:
        stop = k
        duration = time[stop - 1] - time[start]

        if min_dt <= duration <= max_dt:
            sel[start:stop] = True

            block_time_raw = time[start:stop]
            block_i4_raw = i4[start:stop]

            i_limit = (
                (block_i4_raw < ima) &
                (block_i4_raw > imi)
            )

            block_time = block_time_raw[i_limit]
            block_i4 = block_i4_raw[i_limit]

            if len(block_i4) > 0:
                mean_i4 = block_i4.mean()
                g = mean_i4 / vd

                blocks.append({
                    "pulse": len(blocks) + 1,
                    "mean_time": block_time.mean(),
                    "mean_i4": mean_i4,
                    "g": g,
                    "duration": duration,
                    "n_points": len(block_i4),
                })

        start = None

if start is not None:
    stop = len(base)
    duration = time[stop - 1] - time[start]

    if min_dt <= duration <= max_dt:
        sel[start:stop] = True

        block_time_raw = time[start:stop]
        block_i4_raw = i4[start:stop]

        i_limit = (
            (block_i4_raw < ima) &
            (block_i4_raw > imi)
        )

        block_time = block_time_raw[i_limit]
        block_i4 = block_i4_raw[i_limit]

        if len(block_i4) > 0:
            mean_i4 = block_i4.mean()
            g = mean_i4 / vd

            blocks.append({
                "pulse": len(blocks) + 1,
                "mean_time": block_time.mean(),
                "mean_i4": mean_i4,
                "g": g,
                "duration": duration,
                "n_points": len(block_i4),
            })

block_df = pd.DataFrame(blocks)


print("Gefundene Pulse:", len(block_df))
print(block_df)

from scipy.optimize import least_squares

cut = 50
col = "g"

left = block_df.iloc[:cut]
right = block_df.iloc[cut:]

P_left = np.arange(len(left), dtype=float)
P_right = np.arange(len(right), dtype=float)

y_left_raw = left[col].to_numpy()
y_right_raw = right[col].to_numpy()

def normalize_y(y):
    ymin = np.min(y)
    ymax = np.max(y)
    return (y - ymin) / (ymax - ymin), ymin, ymax

x_left = P_left / P_left.max()
x_right = P_right / P_right.max()

y_left, ymin_left, ymax_left = normalize_y(y_left_raw)
y_right, ymin_right, ymax_right = normalize_y(y_right_raw)

def G_LTP_norm(x, A, C, D):
    return C + D * (1 - np.exp(-x / A))

def G_LTD_norm(x, A, C):
    D = y_right[0] - C
    return C + D * np.exp(-x / A)

def residual_ltp(p):
    A, C, D = p
    return G_LTP_norm(x_left, A, C, D) - y_left

def residual_ltd(p):
    A, C = p
    return G_LTD_norm(x_right, A, C) - y_right

res_left = least_squares(
    residual_ltp,
    x0=[0.5, 0.0, 1.0],
    bounds=([0.01, -0.3, 0.1], [10.0, 0.5, 2.0]),
    loss="soft_l1",
    f_scale=0.05,
    max_nfev=10000
)

res_right = least_squares(
    residual_ltd,
    x0=[0.2, y_right[-1]],
    bounds=([0.01, -0.3], [10.0, 0.5]),
    loss="soft_l1",
    f_scale=0.05,
    max_nfev=10000
)

A_left_norm, C_left, D_left = res_left.x
A_right_norm, C_right = res_right.x

fit_left_norm = G_LTP_norm(x_left, A_left_norm, C_left, D_left)
fit_right_norm = G_LTD_norm(x_right, A_right_norm, C_right)

fit_left = fit_left_norm * (ymax_left - ymin_left) + ymin_left
fit_right = fit_right_norm * (ymax_right - ymin_right) + ymin_right

# A wieder in Pulszahlen umrechnen
A_left = A_left_norm * P_left.max()
A_right = A_right_norm * P_right.max()

Gmin = np.min(y_left_raw)
Gmax = np.max(y_left_raw)

print("A_LTP:", A_left)
print("A_LTD:", A_right)
print("LTP params:", res_left.x)
print("LTD params:", res_right.x)


block_df[["pulse", "g"]].to_csv(out_dir / f"{file}_g_vs_pulse.csv", index=False)
fig, ax = plt.subplots()


ax.plot(time[sel], i4[sel], "-", c="royalblue", alpha=0.2, linewidth=0.7)

ax.plot(
    block_df["mean_time"],
    block_df["mean_i4"],
    "-o",
    c="darkblue",
    linewidth=2.0,
    markersize=3,
)

ax.set_xlabel("time")
ax.set_ylabel("mean i4")

fig.savefig(out_dir / f"{file}.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots()

ax.plot(
    block_df["pulse"],
    block_df["g"],
    "o",
    c="darkblue",
    linewidth=2.0,
    markersize=3,
    label="mesasured"
)

ax.set_xlabel("pulse number")
ax.set_ylabel("g [S]")
ax.plot(left.index, fit_left, label=f"LTP Fit, A={A_left:.2f}")
plt.scatter(left.index, fit_left, color="red", s=30)
ax.plot(right.index, fit_right, label=f"LTD Fit, A={A_right:.2f}")
plt.scatter(right.index, fit_right, color="red", s=30)
ax.legend()
fig.savefig(out_dir / f"{file}_g_vs_pulse.png", dpi=300)
matplot2tikz.save(out_dir / f"{file}_g_vs_pulse.tex", figure=fig)
plt.close(fig)

txt_path = out_dir / f"{file}_fit_results.txt"
def alp (A):
    return (1.726/(A+0.162))
Alp_LTP = alp(A_left)
Alp_LTD = alp(A_right)
with open(txt_path, "w") as f:
    f.write(f"Gmin: {Gmin}\n")
    f.write(f"Gmax: {Gmax}\n")
    f.write(f"A_LTP: {A_left}\n")
    f.write(f"A_LTD: {A_right}\n")
    f.write(f"Alp_LTP: {Alp_LTP}\n")
    f.write(f"Alp_LTD: {Alp_LTD}\n")
    f.write(f"beg {beg}\n end {end}\n vread_th {vread_th}\n vread_thp {vread_thp}\n vd {vd}\n min_dt {min_dt}\n max_dt {max_dt}\n ima {ima}\n imi {imi}\n t0 {t0}\n ")
