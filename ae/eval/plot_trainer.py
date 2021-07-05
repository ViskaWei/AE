import matplotlib.pyplot as plt

class Eval(object):
    def __init__(self, wave, tt):
        self.org=tt.flux_org
        self.rec=tt.flux_rec
        self.wave=wave
        self.abs_err= tt.abs_err
    
    def get_abs_err(self):
        return abs(self.org - self.rec)
    
    def plot_flux(self, idx, ax=None):
        ax = ax or plt.gca()
        ax.plot(self.wave, self.org[idx], c='k', label='pca')
        ax.plot(self.wave, self.rec[idx], alpha=0.5, c='r', label='rec')
        ax.set_xlim(self.wave[0], self.wave[-1])
        ax.legend(loc=4)
        
    def plot_err(self, idx, ax=None):
        ax = ax or plt.gca()
        ax.plot(self.wave, self.abs_err[idx], c='k', label='abs_err')
        ax.set_xlim(self.wave[0], self.wave[-1])
        ax.legend(loc=1)
        
    def plot_all(self, idx):
        f, axs = plt.subplots(2,1, figsize=(20,10), sharex="all")
        self.plot_flux(idx, ax=axs[1])
        self.plot_err(idx, ax=axs[0])
        