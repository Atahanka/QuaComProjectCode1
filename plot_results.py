import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str = "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_energy_vs_iteration(
    df: pd.DataFrame,
    p0: float = 0.003,
    shot_budget: int = 900,
    out_file: str = "energy_vs_iter_p0_0.003_B_900.pdf",
):
    regimes = ["noiseless", "noisy", "zne"]
    plt.figure()

    for regime in regimes:
        if regime == "noiseless":
            df_sel = df[(df["regime"] == regime) & (df["shot_budget"] == shot_budget)]
        else:
            df_sel = df[
                (df["regime"] == regime)
                & (df["shot_budget"] == shot_budget)
                & (np.isclose(df["p0"], p0))
            ]

        if df_sel.empty:
            continue

        grouped = df_sel.groupby("iteration")["energy_best"]
        mean_energy = grouped.mean()
        std_energy = grouped.std()

        iterations = mean_energy.index.values

        plt.plot(iterations, mean_energy.values, label=regime)
        plt.fill_between(
            iterations,
            mean_energy.values - std_energy.values,
            mean_energy.values + std_energy.values,
            alpha=0.2,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Best energy so far")
    plt.title(f"Energy vs Iteration (p0={p0}, shots={shot_budget})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_iterations_to_eps_vs_shots(
    df: pd.DataFrame,
    epsilon: float = 0.05,
    out_file: str = "iterations_to_eps_vs_shots.pdf",
):
    regimes = ["noiseless", "noisy", "zne"]
    shot_budgets = sorted(df["shot_budget"].unique())
    exact_ground = df["exact_ground"].iloc[0]

    plt.figure()

    for regime in regimes:
        means = []
        for B in shot_budgets:
            df_B = df[(df["regime"] == regime) & (df["shot_budget"] == B)]

            iters_to_eps = []
            for (p0, seed), df_group in df_B.groupby(["p0", "seed"]):
                df_group = df_group.sort_values("iteration")
                diff = np.abs(df_group["energy_best"] - exact_ground)
                reached = np.where(diff <= epsilon)[0]
                if len(reached) > 0:
                    iters_to_eps.append(int(df_group.iloc[reached[0]]["iteration"]))
                else:
                    max_it = int(df_group["iteration"].max())
                    iters_to_eps.append(max_it + 1)

            if len(iters_to_eps) > 0:
                means.append(np.mean(iters_to_eps))
            else:
                means.append(np.nan)

        plt.plot(shot_budgets, means, marker="o", label=regime)

    plt.xlabel("Shot budget per iteration")
    plt.ylabel(f"Iterations to reach |E - E0| <= {epsilon}")
    plt.title("Convergence speed vs shot budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def plot_final_error_vs_p0(
    df: pd.DataFrame,
    shot_budget: int = 900,
    out_file: str = "final_error_vs_p0_B_900.pdf",
):
    regimes = ["noiseless", "noisy", "zne"]
    exact_ground = df["exact_ground"].iloc[0]

    plt.figure()

    for regime in regimes:
        df_sel = df[(df["regime"] == regime) & (df["shot_budget"] == shot_budget)]
        p0_values = sorted(df_sel["p0"].unique())
        errors_mean = []

        for p0 in p0_values:
            df_p0 = df_sel[df_sel["p0"] == p0]
            last_per_seed = df_p0.groupby("seed").apply(
                lambda g: g.loc[g["iteration"].idxmax()]
            )
            final_errors = last_per_seed["energy_best"] - exact_ground
            errors_mean.append(np.mean(final_errors))

        plt.plot(p0_values, errors_mean, marker="o", label=regime)

    plt.xlabel("Base error rate p0")
    plt.ylabel("Mean final energy error (E_final - E_ground)")
    plt.title(f"Final error vs base error rate (shots={shot_budget})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")


def main():
    df = load_results("results.csv")
    plot_energy_vs_iteration(df, p0=0.003, shot_budget=900)
    plot_iterations_to_eps_vs_shots(df, epsilon=0.05)
    plot_final_error_vs_p0(df, shot_budget=900)


if __name__ == "__main__":
    main()
