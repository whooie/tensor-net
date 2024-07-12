from pathlib import Path
from typing import Callable
import lmfit
import numpy as np
import whooie.pyplotdefs as pd

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

def model_ar(
    params: lmfit.Parameters,
    x: np.ndarray[float, 1],
) -> np.ndarray:
    return params["a"].value * np.ones(x.shape)

def model_vol(
    params: lmfit.Parameters,
    x: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    a = params["a"].value
    b = params["b"].value
    c = params["c"].value
    return a * np.log(x) + b * x + c

def model_crit(
    params: lmfit.Parameters,
    x: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    a = params["a"].value
    b = params["b"].value
    return a * np.log(x) + b

def residuals(
    params: lmfit.Parameters,
    model: Callable[[lmfit.Parameters, np.ndarray], np.ndarray],
    x: np.ndarray[float, 1],
    y: np.ndarray[float, 1],
    err: np.ndarray[float, 1],
) -> np.ndarray[float, 1]:
    m = model(params, x)
    return ((m - y) / err)**2

def corr(x: np.ndarray[float, 1], y: np.ndarray[float, 1]) -> float:
    return np.corrcoef(np.array([x, y]))[0, 1]

def newcorr(x: np.ndarray[float, 1], y: np.ndarray[float, 1]) -> float:
    # see arXiv:1909.10140
    # assumes y is not constant
    sortx = np.argsort(x)
    xp = x[sortx]
    yp = y[sortx]
    n = len(y)
    ranks = np.array([
        sum(1 if yp[k] < yp[i] else 0 for i in range(n) for k in range(i))
    ])
    return (
        1
        - 3 * sum(abs(rkp1 - rk) for (rkp1, rk) in zip(ranks[1:], ranks[:-1]))
        / (n**2 - 1)
    )

def correlations(
    size: np.ndarray[float, 1],
    entropy: np.ndarray[float, 1],
    entropy_err: np.ndarray[float, 1],
) -> tuple[float, float, float]:
    params_ar = lmfit.Parameters()
    params_ar.add("a", value=entropy.mean(), min=0.0)
    fit_ar = lmfit.minimize(
        residuals,
        params_ar,
        args=(model_ar, size, entropy, entropy_err),
    )
    s_ar = model_ar(fit_ar.params, size)
    # c_ar = corr(entropy, s_ar)
    # c_ar = newcorr(s_ar, entropy)
    c_ar = residuals(fit_ar.params, model_ar, size, entropy, entropy_err).sum()

    params_vol = lmfit.Parameters()
    params_vol.add("a", value=1.0, min=0.0)
    params_vol.add("b", value=1.0, min=0.0)
    params_vol.add("c", value=0.0, vary=False)
    fit_vol = lmfit.minimize(
        residuals,
        params_vol,
        args=(model_vol, size, entropy, entropy_err),
    )
    s_vol = model_vol(fit_vol.params, size)
    # c_vol = corr(entropy, s_vol)
    # c_vol = newcorr(s_vol, entropy)
    c_vol = residuals(fit_vol.params, model_vol, size, entropy, entropy_err).sum()

    params_crit = lmfit.Parameters()
    params_crit.add("a", value=1.0, min=0.0)
    params_crit.add("b", value=0.0, vary=False)
    fit_crit = lmfit.minimize(
        residuals,
        params_crit,
        args=(model_crit, size, entropy, entropy_err),
    )
    s_crit = model_crit(fit_crit.params, size)
    # c_crit = corr(entropy, s_crit)
    # c_crit = newcorr(s_crit, entropy)
    c_crit = residuals(fit_crit.params, model_crit, size, entropy, entropy_err).sum()

    # (
    #     pd.Plotter()
    #     .plot(size, s_ar, marker="o", color="C0")
    #     .plot(size, s_vol, marker="o", color="C1")
    #     .plot(size, s_crit, marker="o", color="C3")
    #     .plot(size, entropy, marker="o", color="k")
    #     .set_title(f"{c_ar = }\n{c_vol = }\n{c_crit = }", fontsize="xx-small")
    #     .show()
    #     .close()
    # )

    return (c_ar, c_vol, c_crit)

s_err = (s_std_p + s_std_m) / 2.0
correlations = np.array([
    correlations(size, mean, std)
    for (mean, std) in zip(s_mean, s_err)
]).T

P = pd.Plotter()
it = enumerate(zip(p_meas, s_mean, s_std_p, s_std_m))
for (k, (p_k, mean_k, std_p_k, std_m_k)) in it:
    P.errorbar(
        size, mean_k, np.array([std_m_k, std_p_k]),
        marker="o", linestyle="-", color=f"C{k % 10}",
        label=f"$p = {p_k: .3f}$",
    )
(
    P
    .ggrid()
    .legend(fontsize=4.0)
    .legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize="xx-small",
        framealpha=1.0,
    )
    .set_xlabel("System size")
    .set_ylabel("Entanglement entropy")
    .savefig(outdir.joinpath("phase_transition.png"))
)

# (
#     pd.Plotter()
#     .plot(
#         p_meas, correlations[0, :],
#         marker="o", linestyle="-", color="C0",
#         label="Area law",
#     )
#     .plot(
#         p_meas, correlations[1, :],
#         marker="o", linestyle="-", color="C1",
#         label="Volume law",
#     )
#     .plot(
#         p_meas, correlations[2, :],
#         marker="o", linestyle="-", color="C3",
#         label="Critical",
#     )
#     .ggrid()
#     .legend(fontsize="xx-small")
#     .set_xlabel("$p$")
#     .set_ylabel("$\\chi^2$")
#     .savefig(outdir.joinpath("phase_transition_correlations.png"))
# )

# pd.show()

