import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

def get_svs(s0, svds):
    s1s=[s0]
    v1s = []
    for svd in svds:
        s1 = svd.singular_values_
        v1 = svd.components_.transpose()
        s1s.append(s1)
        v1s.append(v1)
    return s1s, v1s

def get_svs_all(s0, v0, svds):
    s1s=[s0]
    v1s = []
    s1_diffs=[]
    v1_diffs = []
    for svd in svds:
        s1 = svd.singular_values_
        v1 = svd.components_.transpose()

        s1s.append(s1)
        v1s.append(v1)
    return s1s, v1s


def eval_svd(u, s, v, label = ['U', 'V']):
    f, axs = plt.subplots(1, 3, figsize = (8,4), facecolor='w')
    plot_svd_s([s], ax=axs[1], labels=['FULL SVD'])
    
    eval_mat(u, ax = axs[0], f=f, label=label[0], loc='left')
    eval_mat(v, ax = axs[2], f=f, label=label[1])

def eval_mat(mat, ax=None, f=None, label='', loc='right', r=None, vmin=None, vmax=None):
    if ax is None: ax = plt.gca()
    if r is not None:
        vmin = np.quantile(mat, r)
        vmax = np.quantile(mat, 1-r)
        ss = ax.matshow(mat, vmin=vmin, vmax = vmax, aspect='auto')
    else:
        ss = ax.matshow(mat, vmin=vmin, vmax = vmax, aspect='auto')

    ax.set_xlabel(label)
    if f is not None: 
        f.colorbar(ss, ax=[ax], location=loc)


def eval_cov(cov, usv=None):
    if usv is None:
        u0, s0, v0 = np.linalg.svd(cov, full_matrices=True)
        v0= v0.T
        print(np.quantile(cov, 0.95))
    else:
        u0, s0, v0 = usv
    print('plotting')
    f, axs = plt.subplots(2, 2, figsize = (16,16), facecolor='w')
    plot_svd_s([s0], ax=axs[0][0], labels=['FULL SVD'])
    eval_mat(cov, ax = axs[0][1], f=f, label='cov')
    eval_mat(u0, ax = axs[1][0], f=f, label='U', loc='left')
    eval_mat(v0, ax = axs[1][1], f=f, label='V')
    if usv is None:
        return u0, s0, v0


def plot_svd_timing(sList, ts, ax=None):
    if ax is None: ax = plt.gca()
    ax.plot(sList, ts, 'ro-')

# def plot_svd_s(SSList, ax=None):
#     if ax is None: ax = plt.gca()
    
#     for ss in SSList:
#         ss0 = ss / np.sum(ss) 
#         ax.plot(np.arange(len(ss0)), ss0, 'o-')

#     ax.set_xscale('log')
#     ax.set_yscale('log')

#     ax.set_xlabel('Log Singular Value Ratio')
#     ax.set_ylabel('Log Pixel Size')    
# #     ax.legend()


def plot_svd_s(SSList, labels = None, ax=None):
    if ax is None: f, ax = plt.subplots(1, 1, facecolor='w')
    if labels is None: labels = np.arange(len(SSList))
    colors = ['k', 'r','orange','g','b','purple']
    for ii, ss0 in enumerate(SSList):
        ss = ss0[1:]
        # ss0 = ss / np.sum(ss) 
        ss0 =  1 - np.cumsum(ss) / np.sum(ss)
        ax.plot(np.arange(len(ss0)), ss0, 'o-',label = labels[ii], color=colors[ii], ms=1, lw=0.8)
        
    ax.axhline(1e-4, c = "maroon", linestyle = ":", label='1%')   
    ax.axhline(1e-6, c = "maroon", linestyle = ":", label='0.1%')   

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel('Log Singular Value Ratio')
    ax.set_xlabel('Log Rank')    
    # ax.hline
    ax.grid()
    if labels is not None: ax.legend()


def plot_svd_sdiff(SSList, labels = None, ax=None):
    if ax is None: ax = plt.gca()
    if labels is None: labels = np.arange(len(SSList))
    ax.set_xscale('log')
    colors = ['k', 'r','orange','g','b','purple']


    for ii, ss0 in enumerate(SSList):
        ss = ss0[1:]
        if np.sum(ss) == 0: 
            ss0 = ss
        else:
            # ss0 = ss / np.sum(ss) 
            ss0 = 1 - np.cumsum(ss) / np.sum(ss)
        ax.plot(np.arange(len(ss0)), ss0, label = labels[ii],color=colors[ii])

    # ax.set_yscale('log')

    ax.set_ylabel('|$\delta$ Singular Valur Error|')    
    ax.set_xlabel('Log Rank')
    ax.grid()

    if labels is not None: ax.legend()


