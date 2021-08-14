import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import tensorflow as tf


def get_s_from_ww(w):
    N, M = w.shape
    ta = True if N > M else False
    tb = not ta
    cov = tf.linalg.matmul(w, w, transpose_a=ta, transpose_b=tb)
    # print(cov.shape)
    s = tf.linalg.svd(cov, compute_uv=False)
    return s, cov

def plot_ww(model0, model1, cmap='hot'):
    svs0, covs0 = analyze(model0.ae)
    svs1, covs1 = analyze(model1.ae)
    N_layers = len(svs0)
    f, axs = plt.subplots(N_layers, 4, figsize=(20, 5*N_layers))
    for ii in range(N_layers):
        ax = axs[ii]
        sv0 = svs0[ii].numpy()[1:]
        sv1 = svs1[ii].numpy()[1:]
        cov0, cov1 = abs(covs0[ii]), abs(covs1[ii])
        plot_s_hist(sv0, ax=ax[0], label='init', c='grey', alpha=1)
        plot_s_hist(sv0, ax=ax[1], label='init', c='grey', alpha=1)
        plot_s_hist(sv1, ax=ax[1], label='trained', c='r', alpha=0.5)
        ax[0].legend()
        ax[1].legend()

        ax[-2].matshow(cov0, cmap=cmap)
        ax[-1].matshow(cov1, cmap=cmap)

def plot_ww0(svs0, covs0, model1, cmap='hot'):
    svs1, covs1 = analyze(model1.ae)
    N_layers = len(svs0)
    f, axs = plt.subplots(N_layers, 4, figsize=(20, 5*N_layers))
    for ii in range(N_layers):
        ax = axs[ii]
        sv0 = svs0[ii].numpy()[1:]
        sv1 = svs1[ii].numpy()[1:]
        cov0, cov1 = abs(covs0[ii]), abs(covs1[ii])
        plot_s_hist(sv0, ax=ax[0], label='init', c='grey', alpha=1)
        plot_s_hist(sv0, ax=ax[1], label='init', c='grey', alpha=1)
        plot_s_hist(sv1, ax=ax[1], label='trained', c='r', alpha=0.5)
        ax[0].legend()
        ax[1].legend()

        ax[-2].matshow(cov0, cmap=cmap)
        ax[-1].matshow(cov1, cmap=cmap)


def plot_progress(svs0, covs0, model1, cmap='hot'):
    svs1, covs1 = analyze(model1.ae)
    N_layers = len(svs0)
    f, axs = plt.subplots(N_layers, 4, figsize=(20, 5*N_layers))
    for ii in range(N_layers):
        ax = axs[ii]
        sv0 = svs0[ii].numpy()[1:]
        sv1 = svs1[ii].numpy()[1:]
        cov0, cov1 = -abs(covs0[ii]), -abs(covs1[ii])
        plot_s_hist(sv0, ax=ax[0], label='init', c='grey', alpha=1)
        plot_s_hist(sv0, ax=ax[1], label='init', c='grey', alpha=1)
        plot_s_hist(sv1, ax=ax[1], label='trained', c='r', alpha=0.5)
        ax[0].legend()
        ax[1].legend()

        ax[-2].matshow(cov0, cmap=cmap, vmax=0)
        ax[-1].matshow(cov1, cmap=cmap, vmax=0)



def plot_s_hist(s, ax=None, label=None, c=None, alpha=None):
    if ax is None: ax = plt.gca()
    _= ax.hist(s, bins=100, density=True, log=True, label=label, color=c, alpha=alpha)

def analyze(model):
    svs, covs = [], []
    ws = [weight for layer in model.layers for weight in layer.weights if 'kernel' in weight.name]
    for w in ws:
        s, cov = get_s_from_ww(w)
        svs.append(s)
        covs.append(cov)        
    return svs, covs

def plot_esd(svs_init, svs_train=None):
    N = len(svs_init)
    n_row = N//2
    f, axs = plt.subplots(n_row, 2, figsize=(20, n_row*5))
    for r in range(n_row):
        ii = r * 2
        
        ax = axs if n_row == 1 else axs[r]
        _= ax[0].hist(svs_init[ii], bins=100, density=True, log=True, label=['init'], color='grey')
        _= ax[1].hist(svs_init[ii+1], bins=100, density=True, log=True, label=['init'], color='grey')  

        if svs_train is not None:
            _= ax[0].hist(svs_train[ii], bins=100, density=True, log=True, label=['encod'], color='r', alpha=0.5)
            _= ax[1].hist(svs_train[ii+1], bins=100, density=True, log=True, label=['decod'], color='r', alpha=0.5)  

    for ax in axs: 
        ax.legend()
        # ax.set_xscale('log')

def ww_4096(model, rds, plot=1):
    svs = get_svs(model)
    if plot: plot_ww_s(rds, svs, rd_labels=['rd', 'PCA'], ae_labels=['encod','decod'])
    return svs

def get_svs(model, svs=[]):
    for layer in model.ae.layers:
        if len(layer.weights) >= 1:
    #         print(layer.weights)
            mat=layer.weights[0].value().numpy()
            sv = np.linalg.svd(mat, compute_uv=False)    
            svs.append(sv)
    return svs
    


def ww(model, plot=1):
    rds, svs = [], []
    for layer in model.encoder.layers:
        if len(layer.weights) >= 1:
    #         print(layer.weights)
            mat=layer.weights[0].value().numpy()
            rd = np.random.rand(*mat.shape)
            rd_sv = np.linalg.svd(rd, compute_uv=False)   
            rds.append(rd_sv)
            sv = np.linalg.svd(mat, compute_uv=False)    
            svs.append(sv)
    if plot: plot_ww_s(rds, svs)
    return rds, svs




def plot_ww_s(rds, svs, rd_labels=None, ae_labels = None, ax=None):
    if ax is None: f, ax = plt.subplots(1, 1, facecolor='w')
    if rd_labels is None:
        rd_labels = ['rd'+str(idx) for idx in range(len(rds))]
    if ae_labels is None: 
        ae_labels = ['ae'+str(idx) for idx in range(len(svs))]
    colors = ['r','orange','g','b','purple']

    for ii, ss in enumerate(rds):
        ss0 =  1 - np.cumsum(ss) / np.sum(ss)
        ax.plot(np.arange(len(ss0)), ss0, 'o-',label = rd_labels[ii], color='k', ms=1, lw=0.6)

    for ii, ss in enumerate(svs):
        ss0 =  1 - np.cumsum(ss) / np.sum(ss)
        ax.plot(np.arange(len(ss0)), ss0, 'o-',label = ae_labels[ii], color=colors[ii], ms=1, lw=0.8)
        
    # ax.axhline(1e-4, c = "maroon", linestyle = ":", label='1%')   
    # ax.axhline(1e-6, c = "maroon", linestyle = ":", label='0.1%')   

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel('Log Singular Value Ratio')
    ax.set_xlabel('Log Rank')    
    # ax.hline
    ax.grid()
    ax.legend()

def get_norm(x, axis = None):
    # if axis = None
    xm, xstd = x.mean(axis), x.std(axis)
    print(f'mean {xm} | std {xstd} | shape {xm.shape}')
    xnorm = (x - xm) / (3 * xstd)
    print(f'min {xnorm.min()} | max {xnorm.max()}')
    return xnorm 

def get_min_max_norm(x):
    xmin, xmax = x.min(), x.max()
    print(f'min {xmin} | max {xmax}')
    xnorm = (x - xmin ) / (xmax - xmin)
    print(f'mean {xnorm.mean()} | std {xnorm.std()}')
    return xnorm


def get_errs(x, pred, plot_n=None):
    mse_err = np.median(np.sqrt(np.sum((x - pred)**2, axis = 1)))
    mae_err = np.median(np.sum(abs(x - pred), axis = 1))
    print(f'mse: {mse_err} | mae: {mae_err} | ')
    if plot_n is not None: plot_rec(x, pred, N= plot_n, label='pred')
    return mse_err, mae_err
    
def get_pca_errs(u_train, pca_test, latent_dim, plot_n=None):
    U_keep = u_train[:, :int(latent_dim)]
    pca_pred = (pca_test.dot(U_keep)).dot(U_keep.T)
    mse_err, mae_err = get_errs(pca_test, pca_pred, plot_n=plot_n)
    return pca_pred, mse_err, mae_err

def get_ae_errs(x_test, m, plot_n=None):
    ae_pred = m.ae.predict(x_test)
    mse_err, mae_err = get_errs(x_test, ae_pred, plot_n=plot_n)
    return ae_pred, mse_err, mae_err


def prepro(flux, minmax=1):
    assert not np.any(np.isnan(flux))
    log_flux= np.log(flux)    
    norm_flux = -get_norm(log_flux, axis = 0)  
    if minmax: norm_flux = get_min_max_norm(norm_flux)    
    print(norm_flux.shape)
    return norm_flux


def plot_rec(x_test, x_rec, N=5, label='rec'):
    f, axs = plt.subplots(N, 2, figsize=(16, 2 * N), squeeze=False, sharex="all")
    for i in range(N):
        x, y = x_test[i], x_rec[i]
        axs[i , 0].plot(x, lw=0.3, c = 'k')
        axs[i , 0].plot(y, lw=0.3, c= 'r', label = label)
        error = x - y
        mse = np.sqrt(np.sum(error**2)).round(2)
        axs[i , 1].plot(error, lw=0.6, label=f'mse = {mse}', c='k')
    for ax in axs.flatten():
        ax.legend()

def plot_pca_rec(x_test, x_rec, N=5, label='rec'):
    f, axs = plt.subplots(N, 2, figsize=(16, 2 * N), squeeze=False, sharex="all")
    for i in range(N):
        x, y = x_test[i], x_rec[i]
        axs[i , 0].plot(x, lw=0.3, c = 'k')
        axs[i , 0].plot(y, lw=0.3, c= 'r', label = 'PCA')
        error = x - y
        mse = np.sqrt(np.sum(error**2)).round(2)
        axs[i , 1].plot(error, lw=0.6, label=f'mse = {mse}', c='k')
    for ax in axs.flatten():
        ax.legend()

def plot_ae_rec(x_test, ae, N=5):
    f, axs = plt.subplots(N, 2, figsize=(16, 2 * N), squeeze=False, sharex="all")
    for i in range(N):
        test = x_test[i]
        axs[i , 0].plot(test, lw=0.3, c = 'k')
        ae_out =  ae.predict(test.reshape(-1, 4096))[0]
        axs[i , 0].plot(ae_out, lw=0.3, c= 'r', label='AE')
        error = test - ae_out
        mse = np.sqrt(np.sum(error**2)).round(2)
        axs[i , 1].plot(error, lw=0.6, label=f'mse = {mse}', c='k')
    for ax in axs.flatten():
        ax.legend()