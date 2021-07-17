import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_err(org, rec):
    f, axs =  plt.subplots(8, 4,figsize=(20, 40))
    for idx, ax in enumerate(axs.flatten()):
        ax.scatter(org[:, idx],rec[:, idx], s=0.1, color  = 'r')
        ax.set_xlabel(f"PC - {idx + 1}")
        ax.set_ylabel(f"AE - {idx + 1} ")
        ax.grid(1)

def plot_s(s, ax=None, xlog=0, ylog=1):
    ax = ax or plt.gca()
    ss =  1 - np.cumsum(s) / np.sum(s)
    ax.plot(np.arange(len(ss)), ss, 'o-', color='r', ms=1, lw=0.6)

    if ylog: 
        ax.set_yscale('log')
        ax.set_ylabel('Log Singular Value Ratio')
    else:
        ax.set_ylabel('Singular Value Ratio')

    if xlog:
        ax.set_xscale('log')
        ax.set_xlabel('Log Rank')    
    # ax.hline
    ax.grid()
    ax.legend()

def plot_traj(data, i, j, T_eff, ax=None):
    ax = ax or plt.gca()
    ax.scatter(data[:,i], data[:,j], c=T_eff, cmap = "Spectral")
    ax.set_xlabel(f"Latent {i}")
    ax.set_ylabel(f"Latent {j}")

def plot_latent_on(dfen, para):
    sns.pairplot(
        dfen,
        x_vars=[f"e{i}" for i in range(1,6)],
        y_vars=[f"e{i}" for i in range(1,6)],
        hue=para,
        plot_kws=dict(marker="o", s=2, edgecolor="none"),
        diag_kws=dict(fill=False),
        palette="Spectral"
    )