import numpy as np
import whooie.pyplotdefs as pd

# N = 15
# p = 0.20
# mc = 250
# data_020 = np.array([ # Σmin, dt_mean, dt_std+, dt_std-, s_mean, s_std+, s_std-
#     [1e-21,                 3.303, 5.660, 2.119, 10.198,  3.876,  3.173],
#     [1e-19,                 1.916, 4.370, 1.286,  9.695,  4.096,  3.330],
#     [1e-17,                 0.533, 2.497, 0.249,  9.686,  4.172,  3.337],
#     [2.220446049250313e-16, 0.443, 1.644, 0.182,  9.800,  4.391,  3.414],
#     [1e-15,                 0.424, 1.939, 0.158,  9.956,  4.195,  3.154],
#     [1e-13,                 0.443, 2.479, 0.183,  9.844,  3.606,  3.450],
#     [1e-11,                 0.517, 1.918, 0.238,  9.767,  3.832,  2.869],
#     [1e-9,                  0.465, 2.262, 0.200,  9.725,  4.401,  3.189],
#     [1e-7,                  0.465, 2.059, 0.199,  9.790,  4.110,  3.349],
#     [1e-5,                  0.391, 1.572, 0.147,  9.891,  3.894,  3.314],
#     [1e-3,                  0.219, 0.233, 0.023,  9.681,  3.758,  3.305],
# ])
# [sigmin, dt, dt_p, dt_m, sfinal, sfinal_p, sfinal_m] = data_020.T

N = 12
p = 0.05
mc = 150
data_005 = np.array([ # Σmin, dt_mean, dt_std+, dt_std-, s_mean, s_std+, s_std-
    [1e-21,                 8.338, 3.833, 3.160, 30.126,  3.489,  4.862],
    [1e-19,                 7.869, 3.791, 3.369, 30.011,  3.439,  3.945],
    [1e-17,                 4.953, 2.945, 2.503, 29.580,  3.582,  4.786],
    [2.220446049250313e-16, 4.222, 2.624, 1.869, 29.066,  3.673,  4.962],
    [1e-15,                 4.503, 2.928, 1.896, 30.488,  3.764,  3.983],
    [1e-13,                 4.379, 2.623, 2.365, 29.819,  3.780,  4.263],
    [1e-11,                 4.249, 2.876, 2.315, 29.140,  3.469,  3.558],
    [1e-9,                  3.958, 2.626, 2.028, 29.390,  4.144,  4.243],
    [1e-7,                  4.609, 2.993, 1.993, 30.225,  3.902,  4.016],
    [1e-5,                  4.264, 2.780, 2.204, 29.988,  3.311,  3.740],
    [1e-3,                  3.420, 2.481, 1.929, 29.626,  4.202,  5.466],
])
[sigmin, dt, dt_p, dt_m, sfinal, sfinal_p, sfinal_m] = data_005.T

(
    pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)
    [0]
    .fill_between(
        sigmin, dt - dt_m, dt + dt_p,
        linewidth=0.0, color="C0", alpha=0.5,
    )
    .loglog(sigmin, dt, marker="o", linestyle="-", color="C0")
    .ggrid()
    .set_ylabel("Time per run [s]")
    [1]
    .fill_between(
        sigmin, sfinal - sfinal_m, sfinal + sfinal_p,
        linewidth=0.0, color="C1", alpha=0.5,
    )
    .semilogx(sigmin, sfinal, marker="o", linestyle="-", color="C1")
    .ggrid()
    .set_ylabel("Final entropy")
    .set_xlabel("$\Sigma_\\mathregular{min}$")
    .suptitle(f"{N = }; {p = }")
    .savefig("output/compression_test.png")
)

