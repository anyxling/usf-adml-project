import pandas as pd


def clean_parent(path):
    """Cleans the Parent DataFrame."""
    # import df
    df = pd.read_csv(path)

    # pivot data
    df1_pivoted = df.pivot(index="gen1_id", columns="age", values="SHgt_cm")

    missing_df = pd.DataFrame(df1_pivoted.isna().sum(axis=1))
    missing_df_sorted = missing_df.sort_values(by=0, ascending=False)
    missing_df_sorted[missing_df_sorted[0] >= 5]

    # dropping records with 5 or more missing heights
    missing_5 = list(missing_df_sorted[missing_df_sorted[0] >= 5].index)

    df1_less_5_missing = df[~df.gen1_id.isin(missing_5)]

    # interpolation
    df1_fixing = df1_less_5_missing.pivot(
        index="gen1_id", columns="age", values="SHgt_cm"
    )
    df1_fixing.loc[:, 0.10] = df1_fixing.loc[:, 0.10].fillna(
        df1_fixing.loc[:, 0.10].mean()
    )
    df_wide = df1_fixing.interpolate(method="linear", axis=1)

    # make into long format
    df1_wide = df_wide.copy()
    df1_long = df1_wide.reset_index()
    df1_long = df1_long.melt(id_vars=["gen1_id"], var_name="age", value_name="height")
    df1_long = df1_long.sort_values(by=["gen1_id", "age"])
    df_long = df1_long.reset_index(drop=True)

    return df_wide, df_long


def clean_child(path):
    """Cleans the Child DataFrame."""
    # importing df
    df = pd.read_csv(path)

    df.loc[df.study_parent_id_new == 636, "study_parent_sex"] = "father"
    df.loc[df.study_parent_id_new == 482, "study_parent_sex"] = "father"
    # no way of confirming the next two, but I think it should be father
    df.loc[df.study_parent_id_new == 668, "study_parent_sex"] = "father"
    df.loc[df.study_parent_id_new == 724, "study_parent_sex"] = "father"

    df.loc[:, "study_parent_sex"] = df.loc[:, "study_parent_sex"].replace(
        {"father": "M", "mother": "F"}
    )

    # make dictionaries for later
    child_to_parent_id = dict(zip(df.gen2_id, df.study_parent_id_new))
    parent_to_sex = dict(zip(df.study_parent_id_new, df.study_parent_sex))
    child_to_sex = dict(zip(df.gen2_id, df.sex_assigned_at_birth))

    df_pivoted = df.pivot(index="gen2_id", columns="AgeGr", values="SHgt_cm")

    missing_df = pd.DataFrame(df_pivoted.isna().sum(axis=1))
    missing_df_sorted = missing_df.sort_values(by=0, ascending=False)
    missing_df_sorted[missing_df_sorted[0] >= 5]
    missing_5 = list(missing_df_sorted[missing_df_sorted[0] >= 5].index)
    missing_df_sorted[missing_df_sorted[0] >= 5]

    df_less_5_missing = df[~df.gen2_id.isin(missing_5)]

    df2_fixing = df_less_5_missing.pivot(
        index="gen2_id", columns="AgeGr", values="SHgt_cm"
    )
    df2_fixing.loc[:, 0.10] = df2_fixing.loc[:, 0.10].fillna(
        df2_fixing.loc[:, 0.10].mean()
    )
    df_wide = df2_fixing.interpolate(method="linear", axis=1)

    df2_wide = df_wide.copy()
    df2_long = df2_wide.reset_index()
    df2_long = df2_long.melt(id_vars=["gen2_id"], var_name="age", value_name="height")
    df2_long = df2_long.sort_values(by=["gen2_id", "age"])
    df_long = df2_long.reset_index(drop=True)

    return df_wide, df_long, child_to_parent_id, parent_to_sex, child_to_sex
