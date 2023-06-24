# %%
from DMC import DMC

dmc = DMC(
    n_equ=int(1e4),
    n_sam=int(5e6),
    tau_min=0,
    tau_max=5,
    alpha_0=1,
    tau_0=1,
    use_change_alpha=True,
    use_change_beta=False,
)
dmc.equilibrate()
dmc.sample()
dmc.print_I1(0.5)
dmc.print_I1(1)
dmc.print_I2(0.5)
dmc.print_I2(1)
dmc.plot_hist(0.5)
dmc.plot_hist(1)


# %%
from DMC import DMC

dmc = DMC(
    n_equ=int(1e4),
    n_sam=int(5e6),
    tau_min=0,
    tau_max=5,
    alpha_0=1,
    tau_0=5,
    use_change_alpha=False,
    use_change_beta=True,
)
dmc.equilibrate()
dmc.sample()
dmc.plot_hist(bins=120)


# %%
from DMC import DMC

dmc = DMC(
    n_equ=int(1e4),
    n_sam=int(5e6),
    tau_min=0,
    tau_max=5,
    alpha_0=1,
    tau_0=1,
    V=0.5,
    use_change_alpha=True,
    use_change_beta=True,
    use_analytical=True,
)
dmc.equilibrate()
dmc.sample()
dmc.print_I1(0.5)
dmc.print_I1(1)
dmc.print_I2(0.5)
dmc.print_I2(1)
dmc.plot_hist(0.5)
dmc.plot_hist(1)
# %%
from DMC import DMC
from matplotlib import pyplot as plt
import numpy as np

dmc = DMC(
    n_equ=int(1e4),
    n_sam=int(3e6),
    tau_min=0,
    tau_max=5,
    alpha_0=0.4,
    tau_0=1,
    V=0.5,
    use_change_alpha=False,
    use_change_beta=True,
    use_analytical=True,
)
dmc.equilibrate()
dmc.sample()
dmc.plot_hist()
dmc.plot_green_est()
# %%
from DMC import DMC
from matplotlib import pyplot as plt
import numpy as np

dmc = DMC(
    n_equ=int(1e4),
    n_sam=int(3e6),
    tau_min=0,
    tau_max=5,
    alpha_0=1,
    tau_0=1,
    use_change_alpha=False,
    use_change_beta=False,
    use_analytical=True,
)
dmc.equilibrate()
dmc.sample()
dmc.plot_hist()
dmc.plot_green_est()
# %%
