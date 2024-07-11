from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import lmfit
import numpy as np
from whooie.analysis import ExpVal
import whooie.pyplotdefs as pd

@dataclass
class Data:
    p_meas: np.ndarray[float, 1]
    size: np.ndarray[float, 1]
    s_mean: np.ndarray[float, 2]
    s_std_p: np.ndarray[float, 2]
    s_std_m: np.ndarray[float, 2]

def get_data() -> Data:
    outdir = Path("output")
    infile = outdir.joinpath("phase_transition.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.07..0.15_size=5..17.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.12..0.20_size=4..22.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.10..0.20_size=4..20.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.10..0.20_size=5..19.npz")

    data = np.load(str(infile))
    p_meas = data["p_meas"]
    size = data["size"]
    s_mean = data["s_mean"]
    s_std_p = data["s_std_p"]
    s_std_m = data["s_std_m"]
    return Data(p_meas, size, s_mean, s_std_p, s_std_m)

def interp(
    x: np.ndarray[float, 1],
    y: np.ndarray[float, 1],
    x0: float,
) -> float:
    if x0 < x.min() or x0 > x.max():
        raise Exception(
            f"interp: interpolation point {x0:g} is out of bounds"
            f" [{x.min():g}, {x.max():g}]"
        )
    k0 = np.argmin(np.abs(x - x0))
    if x0 == x[k0]:
        return y[k0]
    elif x0 > x[k0]:
        m = (y[k0 + 1] - y[k0]) / (x[k0 + 1] - x[k0])
        return y[k0] + m * (x0 - x[k0])
    elif x0 < x[k0]:
        m = (y[k0] - y[k0 - 1]) / (x[k0] - x[k0 - 1])
        return y[k0 - 1] + m * (x0 - x[k0 - 1])

def gen_scaling(
    p_meas: np.ndarray[float, 1],
    size: np.ndarray[float, 1],
    s_mean: np.ndarray[float, 2],
    pc: float,
    nu: float,
) -> (np.ndarray[float, 2], np.ndarray[float, 2]):
    SIZE, P_MEAS = np.meshgrid(size, p_meas)
    x = ((P_MEAS - pc) * SIZE**(1 / nu))
    s_crit = np.array([interp(p_meas, s_N, pc) for s_N in s_mean.T])
    y = np.array([s_p - s_crit for s_p in s_mean])
    return (x, y)

def residuals(
    params: lmfit.Parameters,
    p_meas: np.ndarray[float, 1],
    size: np.ndarray[float, 1],
    s_mean: np.ndarray[float, 2],
) -> np.ndarray[float, 1]:
    pc = params["pc"].value
    nu = params["nu"].value
    (x, y) = gen_scaling(p_meas, size, s_mean, pc, nu)
    ref_x = x[:, -1]
    ref_y = y[:, -1]
    return np.array([
        (yy - interp(ref_x, ref_y, xx))**2
        for (xx, yy) in zip(x[:, :-1].flatten(), y[:, :-1].flatten())
    ])

def costf(
    params: lmfit.Parameters,
    p_meas: np.ndarray[float, 1],
    size: np.ndarray[float, 1],
    s_mean: np.ndarray[float, 2],
) -> float:
    return residuals(params, p_meas, size, s_mean).sum()

@dataclass
class Fit:
    pc: ExpVal
    nu: ExpVal

    def as_valpair(self) -> (float, float):
        return (self.pc.val, self.nu.val)

def do_fit(data: Data, pc0: float, nu0: float) -> Fit:
    params = lmfit.Parameters()
    params.add("pc", value=pc0, min=data.p_meas.min(), max=data.p_meas.max())
    params.add("nu", value=nu0, min=0.1, max=2.0)
    fit = lmfit.minimize(
        residuals, params, args=(data.p_meas, data.size, data.s_mean))
    pc = ExpVal(fit.params["pc"].value, fit.params["pc"].stderr)
    nu = ExpVal(fit.params["nu"].value, fit.params["nu"].stderr)
    return Fit(pc, nu)

def do_single(data: Data):
    print("single fit")
    fit = do_fit(data, pc0=0.105, nu0=0.564)
    print(f"pc = {fit.pc.value_str()} [{fit.pc.val:.6f} ± {fit.pc.err:.6f}]")
    print(f"nu = {fit.nu.value_str()} [{fit.nu.val:.6f} ± {fit.nu.err:.6f}]")
    cost = eval_costf(data, fit)
    print(f"costf = {cost:.3e}")

    (x, y) = gen_scaling(
        data.p_meas, data.size, data.s_mean, fit.pc.val, fit.nu.val)
    ref_x = x[:, -1]
    ref_y = y[:, -1]
    (
        pd.Plotter()
        .plot(
            x[:, :-1].flatten(), y[:, :-1].flatten(),
            marker="o", linestyle="", color="k",
        )
        .plot(
            ref_x, ref_y,
            marker="o", linestyle="", color="C0",
        )
        .ggrid()
        .set_xlabel("$(p - p_c) N^{1 / \\nu}$")
        .set_ylabel("$S(N, p) - S(N, p_c)$")
        .set_title(f"$p_c = {fit.pc.value_str()}; \\nu = {fit.nu.value_str()};$\ncostf = {cost:.3e}")
        .savefig("output/collapse_single.png")
        # .show()
    )

def eval_costf(
    data: Data,
    fit: Fit,
) -> float:
    params = lmfit.Parameters()
    params.add("pc", value=fit.pc.val)
    params.add("nu", value=fit.nu.val)
    return costf(params, data.p_meas, data.size, data.s_mean)

def eval_costf_bare(
    data: Data,
    pc: float,
    nu: float,
) -> float:
    params = lmfit.Parameters()
    params.add("pc", value=pc)
    params.add("nu", value=nu)
    return costf(params, data.p_meas, data.size, data.s_mean)

def do_spread(data: Data):
    print("fit spread")
    fits = np.array([
        do_fit(data, pc0, nu0)
        for pc0 in np.linspace(data.p_meas.min(), data.p_meas.max(), 50)
        for nu0 in np.linspace(0.5, 2.0, 50)
    ])
    coords = np.array([fit.as_valpair() for fit in fits])
    costfs = np.array([eval_costf(data, fit) for fit in fits])

    hist, pcedges, nuedges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=30, density=True)
    pc = (pcedges[:-1] + pcedges[1:]) / 2.0
    nu = (nuedges[:-1] + nuedges[1:]) / 2.0

    mincost = costfs.argmin()
    minfit = fits[mincost]
    minpc = minfit.pc
    minnu = minfit.nu
    print(f"pc = {minpc.value_str()} [{minpc.val:.6f} ± {minpc.err:.6f}]")
    print(f"nu = {minnu.value_str()} [{minnu.val:.6f} ± {minnu.err:.6f}]")

    (
        pd.Plotter.new_3d()
        .tick_params(pad=0.0, labelsize="x-small")
        .plot(
            coords[:, 0], coords[:, 1], costfs,
            marker="o", linestyle="", color="k",
        )
        .set_xlabel("$p_c$", labelpad=0.0)
        .set_ylabel("$\\nu$", labelpad=0.0)
        .set_zlabel("Collapse distance", labelpad=0.0)
        .savefig("output/collapse_distance.png")
        # .show()
    )

    (
        pd.Plotter()
        .colorplot(pc, nu, hist.T, cmap=pd.colormaps["vibrant"])
        .colorbar()
        .ggrid().grid(False, which="both")
        .set_xlabel("$p_c$")
        .set_ylabel("$\\nu$")
        .set_clabel("Prob. density")
        .savefig("output/collapse_spread.png")
        # .show()
    )

def do_brute_scan(data: Data):
    print("brute-force scan")
    pc = np.linspace(data.p_meas.min(), data.p_meas.max(), 100)
    nu = np.linspace(0.2, 2.0, 100)

    coords = np.array([[pci, nuj] for pci in pc for nuj in nu])
    costfs = np.array([eval_costf_bare(data, *pc_nu) for pc_nu in coords])
    mincost = costfs.argmin()
    minpc = coords[mincost, 0]
    minnu = coords[mincost, 1]
    print(f"pc = {minpc}")
    print(f"nu = {minnu}")
    print(f"costf = {costfs[mincost]}")

    (
        pd.Plotter.new_3d()
        .tick_params(pad=-0.1, labelsize="x-small")
        .plot_trisurf(coords[:, 0], coords[:, 1], np.log10(costfs))
        .set_xlabel("$p_c$", labelpad=-0.05)
        .set_ylabel("$\\nu$", labelpad=-0.05)
        .set_zlabel("Collapse distance (log)", labelpad=-0.05)
        .savefig("output/collapse_distance_brute.png")
        # .show()
    )
    (
        pd.Plotter()
        .colorplot(
            pc, nu, np.log10(costfs.reshape((len(pc), len(nu))).T),
            cmap=pd.colormaps["vibrant"]
        )
        .colorbar()
        .plot([minpc], [minnu], marker="o", linestyle="-", color="r")
        .set_xlabel("$p_c$")
        .set_ylabel("$\\nu$")
        .set_clabel("Collapse distance (log)")
        .ggrid().grid(False, which="both")
        .savefig("output/collapse_distance_brute_2d.png")
        # .show()
    )

def main():
    data = get_data()
    # do_brute_scan(data)
    do_single(data)

if __name__ == "__main__":
    main()

