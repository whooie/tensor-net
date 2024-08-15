from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Self
import lmfit
import numpy as np
from whooie.analysis import ExpVal
import whooie.pyplotdefs as pd

@dataclass
class Data:
    p_meas: np.ndarray[float, 1]
    size: np.ndarray[float, 1]
    s_mean: np.ndarray[float, 1]

    @staticmethod
    def from_file(infile: Path) -> Self:
        data = np.load(str(infile))
        p_meas = data["p_meas"]
        size = data["size"]
        s_mean = data["s_mean"]
        return Data(p_meas, size, s_mean)

def interp(
    x: np.ndarray[float, 1],
    y: np.ndarray[float, 1],
    x0: float,
) -> float:
    # assumes (x, y) pairs are sorted according to x
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
    x = ((P_MEAS - pc) * SIZE ** (1 / nu))
    s_crit = np.array([interp(p_meas, s_N, pc) for s_N in s_mean.T])
    y = np.array([s_p - s_crit for s_p in s_mean])
    return (x, y)

def collapse_distances(
    p_meas: np.ndarray[float, 1],
    size: np.ndarray[float, 1],
    s_mean: np.ndarray[float, 2],
    pc: float,
    nu: float,
) -> np.ndarray[float, 2]:
    (x, y) = gen_scaling(p_meas, size, s_mean, pc, nu)
    ref_x = x[:, -1]
    ref_y = y[:, -1]
    return np.array([
        (yy - interp(ref_x, ref_y, xx)) ** 2
        for (xx, yy) in zip(x[:, :-1].flatten(), y[:, :-1].flatten())
    ]).reshape((s_mean.shape[0], s_mean.shape[1] - 1))

def residuals(
    params: lmfit.Parameters,
    p_meas: np.ndarray[float, 1],
    size: np.ndarray[float, 1],
    s_mean: np.ndarray[float, 2],
) -> np.ndarray[float, 1]:
    pc = params["pc"].value
    nu = params["nu"].value
    return collapse_distances(p_meas, size, s_mean, pc, nu).flatten()

def costf(
    data: Data,
    pc: float,
    nu: float,
) -> float:
    return collapse_distances(
        data.p_meas,
        data.size,
        data.s_mean,
        pc,
        nu,
    ).mean()

@dataclass
class Scan:
    pc: float
    nu: float
    costf: float

def do_scan(data: Data, npoints: int=100) -> Scan:
    pc = np.linspace(data.p_meas.min(), data.p_meas.max(), npoints)
    nu = np.linspace(0.1, 4.0, npoints)

    coords = np.array([[pci, nuj] for pci in pc for nuj in nu])
    costfs = np.array([costf(data, *pc_nu) for pc_nu in coords])
    mincost = costfs.argmin()
    minpc = coords[mincost, 0]
    minnu = coords[mincost, 1]
    mincost = costfs[mincost]

    (
        pd.Plotter()
        .colorplot(
            pc, nu, np.log10(costfs.reshape((npoints, npoints)).T),
            cmap=pd.colormaps["vibrant"],
        )
        .colorbar()
        .plot([minpc], [minnu], marker="o", linestyle="-", color="r")
        .set_xlabel("$p_c$")
        .set_ylabel("$\\nu$")
        .set_clabel("Collapse distance (log)")
        .ggrid().grid(False, which="both")
        .savefig("output/collapse_distance_scan.png")
        .close()
    )

    return Scan(minpc, minnu, mincost)

@dataclass
class Fit:
    pc: ExpVal
    nu: ExpVal
    costf: float

def do_collapse_fit(data: Data, pc0: float, nu0: float) -> Fit:
    params = lmfit.Parameters()
    params.add("pc", value=pc0, min=data.p_meas.min(), max=data.p_meas.max())
    params.add("nu", value=nu0, min=0.1, max=5.0)
    fit = lmfit.minimize(
        residuals, params, args=(data.p_meas, data.size, data.s_mean))
    pc = ExpVal(fit.params["pc"].value, fit.params["pc"].stderr)
    nu = ExpVal(fit.params["nu"].value, fit.params["nu"].stderr)
    cost = costf(data, pc.val, nu.val)

    (x, y) = gen_scaling(data.p_meas, data.size, data.s_mean, pc.val, nu.val)
    ref_x = x[:, -1]
    ref_y = y[:, -1]
    cmap = pd.colormaps["vibrant"]
    nsize = len(data.size)
    colors = [cmap(k / (nsize - 1)) for k in range(nsize)]
    P = pd.Plotter()
    for (xn, yn, c, n) in zip(x[:, :-1].T, y[:, :-1].T, colors, data.size[:-1]):
        P.plot(
            xn, yn,
            marker="o", linestyle="", color=c,
            label=f"$N = {n}$",
        )
    (
        P
        .plot(
            ref_x, ref_y,
            marker="o", linestyle="", color="r",
            label=f"$N = {data.size[-1]}$",
        )
        .ggrid()
        .legend(
            fontsize=4.0,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=1.0,
        )
        .set_xlabel("$(p - p_c) N^{1 / \\nu}$")
        .set_ylabel("$S(N, p) - (N, p_c)$")
        .set_title(
            f"$p_c = {pc.value_str()}; \\nu = {nu.value_str()}$\n"
            f"costf = {cost:.3e}"
        )
        .savefig("output/collapse_result.png")
        .close()
    )

    return Fit(pc, nu, cost)

def main():
    outdir = Path("output")
    infile = outdir.joinpath("phase_transition.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.07..0.15_size=5..17.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.12..0.20_size=4..22.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.10..0.20_size=4..20.npz")
    # infile = outdir.joinpath("phase_transition_p_meas=0.10..0.20_size=5..19.npz")
    # infile = outdir.joinpath("phase_transition_h-s-cx-open_every-prob_small.npz")
    # infile = outdir.joinpath("phase_transition_h-s-cz-open_every-prob_small.npz")
    data = Data.from_file(infile)
    # print("scan")
    # scan = do_scan(data, npoints=200)
    # print(f"pc0 = {scan.pc}")
    # print(f"nu0 = {scan.nu}")
    # print(f"cost = {scan.costf}")
    # print()
    print("fit")
    # fit = do_collapse_fit(data, scan.pc, scan.nu)
    fit = do_collapse_fit(data, 0.26, 1.4)
    print(f"pc = {fit.pc.value_str()}")
    print(f"nu = {fit.nu.value_str()}")
    print(f"cost = {fit.costf:.3e}")

if __name__ == "__main__":
    main()

