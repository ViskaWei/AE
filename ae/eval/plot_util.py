import matplotlib.pyplot as plt

def plot_err(org, rec):
    f, axs =  plt.subplots(8, 4,figsize=(20, 40))
    for idx, ax in enumerate(axs.flatten()):
        ax.scatter(org[:, idx],rec[:, idx], s=0.1, color  = 'r')
        ax.set_xlabel(f"PC - {idx + 1}")
        ax.set_ylabel(f"AE - {idx + 1} ")
        ax.grid(1)