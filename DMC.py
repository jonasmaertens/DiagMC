import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class DMC:
    """
    A class representing the Diagrammatic Monte Carlo (DMC) algorithm for simulating interacting quantum systems.

    Attributes:
    n_equ (int): The number of equilibration steps to perform.
    n_sam (int): The number of samples to generate.
    tau_min (float): The minimum possible imaginary time for the diagrams.
    tau_max (float): The maximum possible imaginary time for the diagrams.
    alpha_0 (float): The initial value of the coupling constant.
    tau_0 (float): The initial value of the imaginary time end point.
    use_change_alpha (bool): Whether to use the change_alpha updater method.
    use_change_beta (bool): Whether to use the add_beta and remove_beta updater methods.
    beta_0 (float): The initial value of the beta coupling constant.
    tau1_0 (float): The initial value of the first interaction vertex time.
    tau2_0 (float): The initial value of the second interaction vertex time.
    V (float): The strength of the interaction.
    n_tau_acc (int): The number of accepted time updates.
    n_tau_rej (int): The number of rejected time updates.
    n_alpha_acc (int): The number of accepted coupling constant updates.
    n_alpha_rej (int): The number of rejected coupling constant updates.
    n_beta_acc (int): The number of accepted beta coupling constant updates.
    n_beta_rej (int): The number of rejected beta coupling constant updates.
    samples (ndarray): An array of shape (n_sam, 2) if use_change_alpha is True, otherwise (n_sam,).
    updaters (list): A list of updater methods to be used during the simulation.
    """

    def __init__(
        self,
        n_equ: int = 1000,
        n_sam: int = 100000,
        tau_min: float = 0,
        tau_max: float = 5,
        alpha_0: float = 1,
        tau_0: float | None = None,
        use_change_alpha: bool = False,
        use_change_beta: bool = True,
        beta_0: float = 0,
        tau1_0: float = 0,
        tau2_0: float = 0,
        V: float = 0.5,
        use_analytical: bool = True,
    ) -> None:
        """
        Initializes the DMC class with the given parameters.

        Parameters:
        n_equ (int): The number of equilibration steps to perform. Defaults to 1000.
        n_sam (int): The number of samples to generate. Defaults to 100_000.
        tau_min (float): The minimum possible imaginary time for the diagrams. Defaults to 0.
        tau_max (float): The maximum possible imaginary time for the diagrams. Defaults to 5.
        alpha_0 (float): The initial value of the coupling constant. Defaults to 1.
        tau_0 (float): The initial value of the imaginary time end point. Defaults to 1.
        use_change_alpha (bool): Whether to use the change_alpha updater method. Defaults to False.
        use_change_beta (bool): Whether to use the add_beta and remove_beta updater methods. Defaults to True.
        beta_0 (float): The initial value of the beta coupling constant. Defaults to 0.
        tau1_0 (float): The initial value of the first interaction vertex time. Defaults to 0.
        tau2_0 (float): The initial value of the second interaction vertex time. Defaults to 0.
        V (float): The strength of the interaction. Defaults to 0.5.
        """
        # check parameters
        if tau_max <= tau_min:
            raise ValueError("tau_max must be greater than tau_min")
        if tau_0 is None:
            tau_0 = tau_min + (tau_max - tau_min) / 2
        if tau_0 < tau_min or tau_0 > tau_max:
            raise ValueError("tau_0 must be between tau_min and tau_max")
        if tau1_0 < tau_min or tau1_0 > tau_max:
            raise ValueError("tau1_0 must be between tau_min and tau_max")
        if tau2_0 < tau_min or tau2_0 > tau_max:
            raise ValueError("tau2_0 must be between tau_min and tau_max")
        # set up random number generator
        self.rng = np.random.default_rng()
        self.uni = lambda a, b: self.rng.random() * (b - a) + a
        # Set up config parameters
        self.n_equ = n_equ
        self.n_sam = n_sam
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.V = V
        self.use_change_beta = use_change_beta
        self.use_change_alpha = use_change_alpha
        self.use_analytical = use_analytical

        # Set up state variables
        self.tau = tau_0
        self.alpha = alpha_0
        self.beta = beta_0
        self.tau1 = tau1_0
        self.tau2 = tau2_0

        # Set up counters
        self.n_tau_acc = 0
        self.n_tau_rej = 0
        self.n_alpha_acc = 0
        self.n_alpha_rej = 0
        self.n_beta_acc = 0
        self.n_beta_rej = 0
        if use_change_alpha:
            self.n_zero_order = {}
        else:
            self.n_zero_order = 0
        # Set up sample array
        # If alpha changes, we need to store tau values for each sample
        self.samples = np.zeros((n_sam, 2) if use_change_alpha else n_sam)

        # Set up updater methods
        self.updaters = [self.change_tau]
        if use_change_alpha:
            self.updaters.append(self.change_alpha)
        if use_change_beta:
            self.updaters.append(self.add_beta)
            self.updaters.append(self.remove_beta)

    def weight(self, tau=None, alpha=None, beta=None, tau1=None, tau2=None) -> float:
        """
        Calculates the weight of the diagram given the current state variables or the provided values.

        Parameters:
        tau (float): The time interval of the diagram. Defaults to the current value of self.tau.
        alpha (float): The coupling constant of the diagram. Defaults to the current value of self.alpha.
        beta (float): The coupling constant of the beta propagator. Defaults to the current value of self.beta.
        tau1 (float): The time of the first interaction vertex. Defaults to the current value of self.tau1.
        tau2 (float): The time of the second interaction vertex. Defaults to the current value of self.tau2.

        Returns:
        float: The weight of the diagram.
        """
        if tau is None:
            tau = self.tau
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
        if tau1 is None:
            tau1 = self.tau1
        if tau2 is None:
            tau2 = self.tau2
        # If the diagram is of first order, the weight is simple
        if beta == 0:
            return np.exp(-alpha * tau)
        # If the diagram is of second order, we need to include
        # two interaction vertices and the beta propagator
        return (
            np.exp(-alpha * tau1)
            * self.V
            * np.exp(-beta * (tau2 - tau1))
            * self.V
            * np.exp(-alpha * (tau - tau2))
        )

    def change_tau(self) -> bool:
        """
        Proposes a new diagram by changing the imaginary time end point tau. The update
        is accepted or rejected using the Metropolis algorithm.

        Returns:
        bool: True if the new tau value is accepted, False otherwise.
        """

        # Propose a new tau value from a uniform distribution.
        # tau must be greater than tau2
        prop_tau = self.uni(self.tau2, self.tau_max)
        # Calculate the Metropolis ratio using the weights
        # of the current and proposed diagrams
        M = self.weight(tau=prop_tau) / self.weight()
        # Accept or reject the new tau value based on the Metropolis ratio
        if M > self.rng.random():
            self.tau = prop_tau
            self.n_tau_acc += 1
            return True
        else:
            self.n_tau_rej += 1
            return False

    def change_alpha(self) -> bool:
        """
        Proposes a new diagram by changing the coupling constant alpha. The update
        is accepted or rejected using the Metropolis algorithm.

        Returns:
        bool: True if the new alpha value is accepted, False otherwise.
        """

        # Propose a new alpha value from {0.5, 1}
        prop_alpha = [0.5, 1][int(self.rng.random() * 2)]
        # Calculate the Metropolis ratio using the weights
        # of the current and proposed diagrams
        M = self.weight(alpha=prop_alpha) / self.weight()
        # Accept or reject the new alpha value based on the Metropolis ratio
        if M > self.rng.random():
            self.alpha = prop_alpha
            self.n_alpha_acc += 1
            return True
        else:
            self.n_alpha_rej += 1
            return False

    def add_beta(self) -> bool:
        """
        Proposes a new diagram by increasing the order to second order.
        The update is accepted or rejected using the Metropolis algorithm.

        Returns:
        bool: True if the new beta value is accepted, False otherwise.
        """

        # If current order is already second order, reject the update
        if self.beta != 0:
            self.n_beta_rej += 1
            return False
        # Propose a new beta value from {0.25, 0.75} with equal probability
        # and two interaction vertices at tau1 and tau2 from a uniform distribution
        # tau_min <= tau1 <= tau2 <= tau <= tau_max
        prop_tau2 = self.uni(self.tau_min, self.tau)
        prop_tau1 = self.uni(self.tau_min, prop_tau2)
        prop_beta = [0.25, 0.75][int(self.rng.random() * 2)]
        # Calculate the Metropolis ratio using the weights of the
        # current and proposed diagrams and the proposal probabilities
        M = self.weight(beta=prop_beta, tau1=prop_tau1, tau2=prop_tau2) / (
            self.weight() / (self.tau - self.tau_min) / (prop_tau2 - self.tau_min) * 0.5
        )
        # Accept or reject the update based on the Metropolis ratio
        if M > self.rng.random():
            self.beta = prop_beta
            self.tau1 = prop_tau1
            self.tau2 = prop_tau2
            self.n_beta_acc += 1
            return True
        else:
            self.n_beta_rej += 1
            return False

    def remove_beta(self) -> bool:
        """
        Proposes a new diagram by decreasing the order to first order.
        The update is accepted or rejected using the Metropolis algorithm.

        Returns:
        bool: True if the new beta value is accepted, False otherwise.
        """

        # If current order is already first order, reject the update
        if self.beta == 0:
            self.n_beta_rej += 1
            return False
        # Calculate the Metropolis ratio using the weights of the
        # current and proposed diagrams and the proposal probabilities
        # This is the inverse of the add_beta() Metropolis ratio
        M = (
            self.weight(beta=0, tau1=0, tau2=0)
            / (self.tau - self.tau_min)
            / (self.tau2 - self.tau_min)
            * 0.5
        ) / self.weight()
        # Accept or reject the update based on the Metropolis ratio
        if M > self.rng.random():
            self.beta = 0
            self.tau1 = 0
            self.tau2 = 0
            self.n_beta_acc += 1
            return True
        else:
            self.n_beta_rej += 1
            return False

    def update(self) -> bool:
        """
        Proposes a new diagram by choosing one of the possible updates
        """

        # Choose a random updater with equal probability
        updater = self.updaters[int(self.rng.random() * len(self.updaters))]  # type: ignore
        # Call the updater and return the result
        return updater()

    def equilibrate(self) -> None:
        """
        Warms up the Markov Chain by performing n_sam updates without sampling
        """
        print(f"Warming up: {self.n_equ} steps")
        # Perform n_equ updates without sampling
        for i in range(self.n_equ):
            self.update()
        # Log the acceptance ratios
        msg = f"Warmup complete: τ acc. ratio of {self.get_tau_acc_ratio():.2f}"
        if self.use_change_alpha:
            msg += f", α acc. ratio of {self.get_alpha_acc_ratio():.2f}"
        if self.use_change_beta:
            msg += f", β acc. ratio of {self.get_beta_acc_ratio():.2f}"
        print(msg)

    def get_alpha_acc_ratio(self) -> float:
        """
        Returns the acceptance ratio of the alpha updates
        """
        if not self.use_change_alpha:
            return 0.0
        return self.n_alpha_acc / (self.n_alpha_acc + self.n_alpha_rej)

    def get_tau_acc_ratio(self) -> float:
        """
        Returns the acceptance ratio of the tau updates
        """
        return self.n_tau_acc / (self.n_tau_acc + self.n_tau_rej)

    def get_beta_acc_ratio(self) -> float:
        """
        Returns the acceptance ratio of the beta updates
        """
        if not self.use_change_beta:
            return 0.0
        return self.n_beta_acc / (self.n_beta_acc + self.n_beta_rej)

    def sample(self) -> None:
        """
        Samples n_sam tau values from the Markov Chain and stores them in self.samples
        """
        print(f"Sampling: {self.n_sam} steps")
        # use tqdm to show a progress bar while sampling n_sam diagrams
        for i in trange(self.n_sam):
            self.update()
            # If alpha changes, the alpha value also needs to be saved
            if self.use_change_alpha:
                self.samples[i, 0] = self.tau
                self.samples[i, 1] = self.alpha
                if self.beta == 0:
                    self.n_zero_order[self.alpha] = self.n_zero_order.get(self.alpha, 0) + 1  # type: ignore
            else:
                self.samples[i] = self.tau
                if self.beta == 0:
                    self.n_zero_order += 1  # type: ignore
        # Log the acceptance ratios
        msg = f"Sampling complete: τ acc. ratio of {self.get_tau_acc_ratio():.2f}"
        if self.use_change_alpha:
            msg += f", α acc. ratio of {self.get_alpha_acc_ratio():.2f}"
        if self.use_change_beta:
            msg += f", β acc. ratio of {self.get_beta_acc_ratio():.2f}"
        print(msg)

    def get_samples(self, alpha: float | None = None) -> np.ndarray:
        """
        Returns the tau samples for a given alpha value

        Parameters:
        alpha (float): The alpha value for which the samples should be returned.
            If None, the current alpha value is used.

        Returns:
        np.ndarray: The tau samples for the given alpha value
        """
        if alpha is None:
            alpha = self.alpha
        if self.use_change_alpha:
            rows = np.where(self.samples[:, 1] == alpha)
            return self.samples[rows, 0][0]
        else:
            return self.samples

    def norm(self, alpha: float | None = None, analytical: bool = True) -> float:
        """
        Returns the normalization constant for a given alpha value

        Parameters:
        alpha (float): The alpha value for which the normalization constant should be returned.
            If None, the current alpha value is used.
        analytical (bool): If True, the analytical expression for the normalization constant is used.

        Returns:
        float: The normalization constant for the given alpha value
        """
        if alpha is None:
            alpha = self.alpha
        norm_zero_order = (
            np.exp(-alpha * self.tau_min) - np.exp(-alpha * self.tau_max)
        ) / alpha
        if not self.use_change_beta:
            return norm_zero_order
        if analytical:
            res = (
                1
                / alpha**2
                * (
                    alpha
                    * (-np.exp(-alpha * self.tau_max) + np.exp(-alpha * self.tau_min))
                    + 1
                    / (0.75 - alpha) ** 2
                    * (
                        -0.75 * np.exp(-alpha * self.tau_max)
                        + 0.75 * np.exp(-alpha * self.tau_min)
                        + alpha
                        * np.exp(-alpha * self.tau_max)
                        * (2 - 0.75 * self.tau_max)
                        + alpha
                        * np.exp(-alpha * self.tau_min)
                        * (-2.0 + 0.75 * self.tau_min)
                        + alpha**2
                        * (
                            -4 / 3 * np.exp(-0.75 * self.tau_max)
                            + 4 / 3 * np.exp(-0.75 * self.tau_min)
                            + np.exp(-alpha * self.tau_max) * self.tau_max
                            - np.exp(-alpha * self.tau_min) * self.tau_min
                        )
                    )
                    * self.V**2
                    + 1
                    / (0.25 - alpha) ** 2
                    * (
                        -0.25 * np.exp(-alpha * self.tau_max)
                        + 0.25 * np.exp(-alpha * self.tau_min)
                        + alpha
                        * np.exp(-alpha * self.tau_max)
                        * (2 - 0.25 * self.tau_max)
                        + alpha
                        * np.exp(-alpha * self.tau_min)
                        * (-2.0 + 0.25 * self.tau_min)
                        + alpha**2
                        * (
                            -4 * np.exp(-0.25 * self.tau_max)
                            + 4 * np.exp(-0.25 * self.tau_min)
                            + np.exp(-alpha * self.tau_max) * self.tau_max
                            - np.exp(-alpha * self.tau_min) * self.tau_min
                        )
                    )
                    * self.V**2
                )
            )
        else:
            if self.use_change_alpha:
                n_sam = np.where(self.samples[:, 1] == alpha)[0].size
                res = norm_zero_order * n_sam / self.n_zero_order[alpha]  # type: ignore
            else:
                res = norm_zero_order * self.n_sam / self.n_zero_order
        return res

    def print_I1(self, alpha: float | None = None) -> None:
        """
        Prints the mean and variance of the integral \int_tau_min^tau_max tau * Q(tau, alpha, [V]) for a given alpha value

        Parameters:
        alpha (float): The alpha value for which the mean and variance should be printed
            If None, the current alpha value is used.

        """
        if alpha is None:
            alpha = self.alpha
        samples = self.get_samples(alpha) * self.norm(
            alpha, analytical=self.use_analytical
        )
        mean, var = self.blocking(samples)
        print(f"I1(α={alpha}) mean: {mean:.5f}, var: {var:.2e}")

    def print_I2(self, alpha: float | None = None) -> None:
        """
        Prints the mean and variance of the integral \int_tau_min^tau_max tau^2 * Q(tau, alpha, [V]) for a given alpha value

        Parameters:
        alpha (float): The alpha value for which the mean and variance should be printed
            If None, the current alpha value is used.
        """
        samples = self.get_samples(alpha) ** 2 * self.norm(
            alpha, analytical=self.use_analytical
        )
        mean, var = self.blocking(samples)
        print(f"I2(α={alpha}) mean: {mean:.5f}, var: {var:.2e}")

    def plot_hist(self, alpha: float | None = None, bins: int = 100) -> None:
        """
        Plots the histogram of the samples for a given alpha value

        Parameters:
        alpha (float): The alpha value for which the histogram should be plotted
            If None, the current alpha value is used.
        bins (int): The number of bins to use for the histogram
        """
        if alpha is None:
            alpha = self.alpha
        samples = self.get_samples(alpha)
        # create normalized histogram
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = np.diff(bin_edges)
        # Scale the histogram by the analytical normalization constant
        norm = self.norm(alpha, analytical=self.use_analytical)
        hist = hist * norm
        # Plot the histogram and the analytical function
        plt.xlabel(r"Imaginary time $\tau$")
        plt.ylabel(
            f"$Q(\\tau,\\alpha={alpha}{f',V={self.V}' if self.use_change_beta else ''})$"
        )
        plt.bar(bin_centers, hist, width=bin_widths, label=r"$\rho(\tau)\cdot Norm$")
        x_vals = np.linspace(0, 5, 1000)
        plt.plot(
            x_vals,
            self.Q_analytical(x_vals, alpha),
            label=f"$Q_{{exact}}(\\tau, \\alpha={alpha}{f',V={self.V}' if self.use_change_beta else ''})$",
            color="red",
        )
        plt.legend()
        plt.show()

    def blocking(self, samples, min_blocks: int = 32) -> tuple[float, float]:
        """
        Performs blocking analysis on the given samples

        Parameters:
        samples (np.ndarray): The samples to perform the blocking analysis on
        min_blocks (int): The minimum number of blocks to use for the blocking analysis

        Returns:
        mean (float): The mean of the samples
        var (float): The blocked variance of the samples
        """
        # Copy the samples to avoid changing the original array
        means = np.copy(samples)
        # Calculate the mean of the samples
        mean = np.mean(means).astype(float)
        # Calculate the iteration steps
        # Minumum block size is 1, maximum is 2**n
        n = np.log2(len(means) // min_blocks).astype(int)
        block_sizes = np.logspace(0, n, n + 1, base=2)
        # Initialize the array for the variances and
        # calculate the non-blocked variance
        vars = np.zeros(n + 1, dtype=float)
        vars[0] = 1 / (len(means) - 1) * (np.mean(means**2) - mean**2)
        # Initialize Rx array
        Rx = np.zeros(n + 1)
        Rx[0] = 1
        # Perform blocking
        for i in range(1, n + 1):
            # Double the block size
            # Make sure the number of blocks is even
            # in order to divide by 2
            if means.size % 2 != 0:
                means = means[:-1]
            means = 0.5 * (means[::2] + means[1::2])
            # Calculate the variance
            n_blocks = means.size
            varXBlock = n_blocks / (n_blocks - 1) * (np.mean(means**2) - mean**2)
            vars[i] = varXBlock / n_blocks
            Rx[i] = block_sizes[i] * varXBlock / vars[0] / samples.size
        # Find plateau in Rx
        plateau_index = np.argmin(np.abs(Rx[1:] - Rx[:-1])[3:]) + 4
        # plot
        fig, ax = plt.subplots()
        ax.set_xscale("log", base=2)
        ax.set_xticks(block_sizes)
        ax.grid()
        ax.set_xlabel("Block size")
        ax.set_ylabel(r"$R_x$")
        ax.plot(block_sizes, Rx)
        ax.scatter(block_sizes[plateau_index], Rx[plateau_index], color="red")
        ax2 = ax.twinx()
        ax2.ticklabel_format(axis="y", style="sci")
        ax2.set_ylabel(r"$\sigma^2$")
        ax2.plot(block_sizes, vars, alpha=0)
        fig.tight_layout()
        plt.show()
        return mean, vars[plateau_index]

    def Q_analytical(
        self, tau: float | np.ndarray, alpha: float | None = None
    ) -> float | np.ndarray:
        """
        Returns the analytically calculated Q function for a given alpha value

        Parameters:
        tau (float | np.ndarray): The imaginary time value(s) for which the Q function should be calculated
        alpha (float): The alpha value for which the Q function should be calculated
            If None, the current alpha value is used.
        
        Returns:
        Q (float | np.ndarray): The Q function for the given alpha value and imaginary time value(s)
        """
        if alpha is None:
            alpha = self.alpha
        V = self.V
        # For first order only the first term is needed
        if not self.use_change_beta:
            return np.exp(-alpha * tau)
        return (
            np.exp(-alpha * tau)
            + (
                np.exp((-0.25 - alpha) * tau)
                * (
                    np.exp(alpha * tau)
                    + np.exp(0.25 * tau) * (-1 + (0.25 - alpha) * tau)
                )
                * V**2
            )
            / (0.25 - alpha) ** 2
            + (
                np.exp((-0.75 - alpha) * tau)
                * (
                    np.exp(alpha * tau)
                    + np.exp(0.75 * tau) * (-1 + (0.75 - alpha) * tau)
                )
                * V**2
            )
            / (0.75 - alpha) ** 2
        )

    def green_estimator(self, tau0: float | np.ndarray, alpha: float | None = None) -> float | np.ndarray:
        """
        Returns the green function estimator for a given tau0
        Gamma is set to 2*min(tau0, 0.5, self.tau_max-tau0)

        Parameters:
        tau0 (float | np.ndarray): The imaginary time value(s) for which the green function estimator should be calculated
        alpha (float): The alpha value for which the green function estimator should be calculated
            If None, the current alpha value is used.

        Returns:
        (float | np.ndarray): The green function estimator for the given alpha value and imaginary time value(s)
        """
        if alpha is None:
            alpha = self.alpha
        tau = self.get_samples(alpha)
        diff = np.subtract.outer(tau0, tau)
        tau0_stacked = np.repeat(tau0, tau.size).reshape(diff.shape)
        mins = np.minimum(np.minimum(tau0_stacked, self.tau_max - tau0_stacked), 0.2)
        res = np.exp(-alpha * (diff)) / (2 * mins)
        res[np.abs(diff) > mins] = 0
        meaned = np.mean(res, axis=1)
        meaned *= self.norm(alpha, analytical=self.use_analytical)
        return meaned

    def plot_green_est(self, alpha: float | None = None) -> None:
        """
        Plots the green function estimator for a given alpha

        Parameters:
        alpha (float): The alpha value for which the green function estimator should be plotted
        """
        if alpha is None:
            alpha = self.alpha
        x = np.linspace(self.tau_min + 0.05, self.tau_max - 0.05, 100)
        plt.xlabel(r"Imaginary time $\tau$")
        plt.ylabel(
            f"$Q(\\tau,\\alpha={alpha}{f',V={self.V}' if self.use_change_beta else ''})$"
        )
        plt.plot(
            x,
            self.green_estimator(x),
            label=f"$Q_{{est}}(\\tau, \\alpha={alpha}{f',V={self.V}' if self.use_change_beta else ''})$",
            color="green",
        )
        plt.plot(
            x,
            self.Q_analytical(x, alpha),
            color="red",
            alpha=0.6,
            label=f"$Q_{{exact}}(\\tau, \\alpha={alpha}{f',V={self.V}' if self.use_change_beta else ''})$",
        )
        plt.legend()
