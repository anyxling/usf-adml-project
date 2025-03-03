import matplotlib.pyplot as plt
import seaborn as sns
from better_velocity import calculate_phv


def plot_velocity_curves(velocity_df, title="Growth Velocity Across Individuals"):
    """
    Plots growth velocity curves for multiple individuals.
    """
    plt.figure(figsize=(10, 6))

    # Plot each individual's velocity curve
    for _, group in velocity_df.groupby("id"):
        plt.plot(group["Age"], group["Velocity"], alpha=0.3)

    # Plot the mean trend
    avg_velocity = velocity_df.groupby("Age")["Velocity"].mean()
    sns.lineplot(
        x=avg_velocity.index,
        y=avg_velocity.values,
        color="red",
        label="Mean Trend",
        linewidth=2,
    )

    plt.xlabel("Age (Years)")
    plt.ylabel("Growth Velocity (cm/year)")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_velocity_curves1(child_wide, start_row, end_row):
    """
    Plots velocity curves for individuals in the specified row range.

    Parameters:
    - child_wide: DataFrame containing height data.
    - start_row: The starting row index (inclusive).
    - end_row: The ending row index (inclusive).
    """
    plt.figure(figsize=(20, 6))

    for idx in child_wide.iloc[
        start_row : end_row + 1, 8:22
    ].index:  # Select range of rows
        age = (
            child_wide.iloc[start_row : end_row + 1, 8:22]
            .columns.astype(float)
            .values
        )
        height = child_wide.iloc[start_row : end_row + 1, 8:22].loc[idx].values
        age_smooth, _, velocity_smooth = calculate_phv(
            age, height
        )  # Compute velocity

        plt.plot(age_smooth, velocity_smooth, label=f"Velocity Curve - {idx}")

    plt.xlabel("Age (years)")
    plt.ylabel("Velocity (cm/year)")
    plt.title("Velocity Curves")
    plt.legend()
    plt.show()
