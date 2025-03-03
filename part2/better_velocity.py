import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema, savgol_filter


def extract_puberty_features(child_wide):
    """
    Computes PHV, Age at PHV, Puberty Onset (TO), and Puberty End using velocity curves.

    - Computes velocity using ages 4-18.
    - Finds PHV, TO, and End only in ages 8-18.

    Returns:
    - puberty_df: DataFrame with PHV, Age_at_PHV, Age_at_TO (onset), and Age_at_End (offset).
    """
    puberty_data = []

    for idx in child_wide.index:
        # Get the full age and height data (ages 4-18)
        age = child_wide.columns.astype(float).values
        height = child_wide.loc[idx].values

        # Interpolation
        f = interp1d(age, height, kind="cubic", fill_value="extrapolate")
        age_smooth = np.linspace(age.min(), age.max(), 1000)  # Keep full range (4-18)
        height_smooth = f(age_smooth)

        # Compute velocity (1st derivative)
        velocity = np.gradient(height_smooth, age_smooth)

        # Smooth using Savitzky-Golay filter
        velocity_smooth = savgol_filter(velocity, window_length=101, polyorder=3)

        # Now focus on ages **8-18** for puberty detection
        valid_range_mask = (age_smooth >= 8) & (age_smooth <= 16)
        valid_ages = age_smooth[valid_range_mask]
        valid_velocities = velocity_smooth[valid_range_mask]

        # Find PHV (maximum velocity within 8-18)
        phv_index = np.argmax(valid_velocities)
        phv = valid_velocities[phv_index]
        age_at_phv = valid_ages[phv_index]

        # Find local minima (puberty onset & end) within 8-18
        min_indices = argrelextrema(valid_velocities, np.less)[0]

        # Onset (TO): Closest local min **before** PHV
        min_before_phv = min_indices[min_indices < phv_index]
        age_at_to = (
            valid_ages[min_before_phv[-1]] if len(min_before_phv) > 0 else np.nan
        )

        # End of puberty: Closest local min **after** PHV
        min_after_phv = min_indices[min_indices > phv_index]
        age_at_end = valid_ages[min_after_phv[0]] if len(min_after_phv) > 0 else np.nan

        # Store results
        puberty_data.append(
            {
                "id": idx,
                "PHV": phv,
                "Age_at_PHV": age_at_phv,
                "Age_at_Onset": age_at_to,
                "Age_at_End": age_at_end,
            }
        )

    # Convert to DataFrame
    puberty_df = pd.DataFrame(puberty_data).set_index("id")

    # Fill missing Age_at_End with 18 if Age_at_PHV is not 18
    puberty_df.loc[
        (puberty_df["Age_at_End"].isna()) & (puberty_df["Age_at_PHV"] < 18),
        "Age_at_End",
    ] = 18

    # Fill missing Age_at_TO with 8 if Age_at_TO is not 8
    puberty_df.loc[
        (puberty_df["Age_at_Onset"].isna()) & (puberty_df["Age_at_PHV"] > 8), "Age_at_Onset"
    ] = 8

    # Make duration column
    puberty_df.loc[:, "Duration"] = puberty_df.Age_at_End - puberty_df.Age_at_Onset

    return puberty_df


def calculate_phv(age, height):
    # Create a smooth interpolation of the data
    f = interp1d(age, height, kind="cubic")
    age_smooth = np.linspace(age.min(), age.max(), 1000)
    height_smooth = f(age_smooth)

    # Calculate the derivative (velocity)
    velocity = np.gradient(height_smooth, age_smooth)

    # Apply Savitzky-Golay filter to smooth the velocity curve
    velocity_smooth = savgol_filter(velocity, window_length=101, polyorder=3)

    return age_smooth, height_smooth, velocity_smooth
