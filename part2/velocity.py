import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev

def fit_spline_with_velocity(df, age_cols, min_age=9, max_age=18, k=3, smoothing=10):
    """
    Fits a B-spline to each individual's growth curve, extracts key growth features, 
    and returns velocity curves for visualization.

    Parameters:
    - df: Wide-format DataFrame (index is gen1_id or gen2_id, columns are heights at ages)
    - age_cols: List of columns representing height at different ages
    - min_age: Minimum age to consider for detecting PHV (default=9)
    - max_age: Maximum age to consider for detecting PHV (default=18)
    - k: Degree of the B-spline (default=3 for cubic)
    - smoothing: Smoothing factor (higher values = smoother curve)

    Returns:
    - features_df: DataFrame with PHV, Age at PHV, and Max Acceleration Age
    - velocity_df: DataFrame with growth velocity curves for plotting
    """
    spline_features = []
    velocity_data = []

    for individual_id, row in df.iterrows():
        # Convert age columns to float and filter only numeric columns
        numeric_ages = np.array([float(col) for col in age_cols if isinstance(col, (int, float))])

        # Apply a rolling mean smoothing before fitting the spline
        heights = row[age_cols].astype(float).rolling(window=3, min_periods=1).mean().values

        # Filter only ages within puberty range
        valid_mask = (numeric_ages >= min_age) & (numeric_ages <= max_age)
        filtered_ages, filtered_heights = numeric_ages[valid_mask], heights[valid_mask]

        # Fit B-spline
        tck = splrep(filtered_ages, filtered_heights, k=k, s=smoothing)

        # Evaluate spline over a dense age range
        age_smooth = np.linspace(filtered_ages.min(), filtered_ages.max(), 100)
        velocity_smooth = splev(age_smooth, tck, der=1)  # First derivative (growth velocity)
        acceleration_smooth = splev(age_smooth, tck, der=2)  # Second derivative (growth acceleration)
        
        # Clip extreme velocities to a reasonable range
        velocity_smooth = np.clip(velocity_smooth, -5, 15)

        # Find PHV (max velocity) **only in puberty range**
        phv_index = np.argmax(velocity_smooth)
        age_at_phv = age_smooth[phv_index]
        phv = velocity_smooth[phv_index]

        # Find puberty onset (max acceleration) **only in puberty range**
        max_accel_index = np.argmax(acceleration_smooth)
        age_at_max_accel = age_smooth[max_accel_index]

        # Store extracted features
        spline_features.append({
            "id": individual_id,
            "PHV": phv,
            "Age_at_PHV": age_at_phv,
            "Max_Accel_Age": age_at_max_accel,
        })

        # Store velocity curve for plotting
        velocity_data.append(pd.DataFrame({
            "id": individual_id,
            "Age": age_smooth,
            "Velocity": velocity_smooth
        }))

    # Create DataFrame for extracted features
    features_df = pd.DataFrame(spline_features).set_index("id")

    # Create DataFrame for velocity curves
    velocity_df = pd.concat(velocity_data, ignore_index=True)

    return features_df, velocity_df