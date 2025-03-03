import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_parent_lowrank_approx(df, var_explained=95):
    pca = PCA()
    pc_scores = pca.fit_transform(df)

    per_var = 100 * pca.explained_variance_ratio_
    cum_variance = np.cumsum(per_var)
    num_components = np.argmax(cum_variance >= var_explained) + 1

    pca_optimal = PCA(n_components=num_components)
    pc_scores_optimal = pca_optimal.fit_transform(df)

    # making pca df
    pc_df = pd.DataFrame(
        pc_scores_optimal,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(num_components)],
    )

    # rank k approx:
    # first approximate the centered versions
    lowrank_approx_without_mean = (
        pc_scores[:, :num_components] @ pca.components_[:num_components, :]
    )

    # reconstruct the data by adding the mean back in
    lowrank_approx = lowrank_approx_without_mean + pca.mean_

    # put into pd.DataFrame
    lowrank_approx_df = pd.DataFrame(
        lowrank_approx.round(5), columns=df.columns, index=df.index
    )

    return lowrank_approx_df, pc_df
