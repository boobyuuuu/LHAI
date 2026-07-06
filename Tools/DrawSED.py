import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def brokenPL(x, pars):

    y1 = pars[0]*pow(x, -pars[2])

    return
    if x<pars[1]:
        return pars[0]*pow(x, -pars[2])
    else:
        return pars[0]*pow(x, -pars[2])


def SEDerr_func(x, pars, parerrs, epiv, functype='LP'):

    if functype == 'LP':
        y = np.sqrt(pow(parerrs[0]/pars[0], 2)+pow(parerrs[1]*np.log(x/epiv), 2)+pow(parerrs[2]*np.log(x/epiv)*np.log(x/epiv), 2))
        return y
    elif functype == 'PEC':
        y = np.sqrt(pow(parerrs[0]/pars[0], 2)+pow(parerrs[1]*np.log(x/epiv), 2)+pow(x/pars[2]/pars[2]*parerrs[2], 2))
        return y
    elif functype == 'PL':
        y = np.sqrt(pow(parerrs[0]/pars[0], 2)+pow(parerrs[1]*np.log(x/epiv), 2))
        return y
    elif functype == 'BPL':
        x1 = x[x<pars[1]]
        x2 = x[x>pars[1]]
        y1 = np.sqrt(pow(parerrs[0]/pars[0], 2)+pow(parerrs[2]*np.log(x1/pars[1]), 2))
        y2 = np.sqrt(pow(parerrs[0]/pars[0], 2)+pow(parerrs[3]*np.log(x2/pars[1]), 2))
        y = np.concatenate((y1, y2), axis=0)
        return y
    else:
        raise ValueError('input error : input functype is not supported')


def SED_func(x, pars, epiv, f0_order, functype='LP'):

    if functype == 'LP':
        y = x*x*pars[0]*f0_order*pow(x/epiv, -pars[1]-pars[2]*np.log(x/epiv))
        return y
    elif functype == 'LP2':
        y = x*x*pars[0]*f0_order*pow(x/epiv, -pars[1]-pars[2]*np.log10(x/epiv))
        return y
    elif functype == 'PEC':
        y = x*x*pars[0]*f0_order*pow(x/epiv, -pars[1])*np.exp(-x/pars[2])
        return y
    elif functype == 'PL':
        y = x*x*pars[0]*f0_order*pow(x/epiv, -pars[1])
        return y
    elif functype == 'BPL':
        x1 = x[x<pars[1]]
        x2 = x[x>pars[1]]
        y1 = x1*x1*pars[0]*f0_order*pow(x1/pars[1], -pars[2])
        y2 = x2*x2*pars[0]*f0_order*pow(x2/pars[1], -pars[3])
        y = np.concatenate((y1, y2), axis=0)
        return y
    else:
        raise ValueError('input error : input functype is not supported')


def SED_ploter(ax, data, f0_order, epiv, pars, parerrs, functype, param_dictW, param_dictK, labelW='', labelK='', linecolor='gray'):

    wcdadata = data[(data['WCDAtag']==1) & (data['ferr']>0)]
    km2adata = data[(data['WCDAtag']==0) & (data['ferr']>0)]
    wcdaul   = data[(data['WCDAtag']==1) & (data['ferr']<=0)]
    km2aul   = data[(data['WCDAtag']==0) & (data['ferr']<=0)]

    if len(wcdadata) != 0:
        ax.errorbar(wcdadata['energy'], wcdadata['flux']*f0_order, yerr=wcdadata['ferr']*f0_order, label=labelW, **param_dictW)
    if len(km2adata) != 0:
        ax.errorbar(km2adata['energy'], km2adata['flux']*f0_order, yerr=km2adata['ferr']*f0_order, label=labelK, **param_dictK)
    if len(wcdaul) != 0:
        ax.errorbar(wcdaul['energy'], wcdaul['flux']*f0_order, yerr=wcdaul['flux']*0.25*f0_order, uplims=1, mfc='none', **param_dictW)
    if len(km2aul) != 0:
        ax.errorbar(km2aul['energy'], km2aul['flux']*f0_order, yerr=km2aul['flux']*0.25*f0_order, uplims=1, mfc='none', **param_dictK)

    ax.set_xscale('log')
    ax.set_xlabel('Energy [ TeV ]')
    ax.set_yscale('log')
    ax.set_ylabel('E$^2$Flux [ TeV cm$^{-2}$s$^{-1}$ ] ')
    ax.legend()

    if functype != 'none':
        npoint = len(data)
        ee = np.logspace(np.log10(data['energy'][0])-0.2, np.log10(data['energy'][npoint-1])+0.2, 100)
        ff = SED_func(ee, pars, epiv, f0_order, functype=functype)
        ax.plot(ee, ff, '-', linewidth=2, color=linecolor)
        fferr = SEDerr_func(ee, pars, parerrs, epiv, functype=functype)
        ax.fill_between(ee, ff-ff*fferr, ff+ff*fferr, alpha=0.1, color=linecolor)

    #return 1


def SED_ploter_AsyErr(ax, data, f0_order, epiv, pars, parerrs, functype, param_dictW, param_dictK, labelW='', labelK='', linecolor='gray'):

    wcdadata = data[(data['WCDAtag']==1) & (data['ferrL']>0)]
    km2adata = data[(data['WCDAtag']==0) & (data['ferrL']>0)]
    wcdaul   = data[(data['WCDAtag']==1) & (data['ferrL']<=0)]
    km2aul   = data[(data['WCDAtag']==0) & (data['ferrL']<=0)]
    wcdaerr  = np.zeros((2, len(wcdadata)))
    wcdaerr[0] = wcdadata['ferrL']
    wcdaerr[1] = wcdadata['ferrU']
    km2aerr  = np.zeros((2, len(km2adata)))
    km2aerr[0] = km2adata['ferrL']
    km2aerr[1] = km2adata['ferrU']

    if len(wcdadata) != 0:
        ax.errorbar(wcdadata['energy'], wcdadata['flux']*f0_order, yerr=wcdaerr*f0_order, label=labelW, **param_dictW)
    if len(km2adata) != 0:
        ax.errorbar(km2adata['energy'], km2adata['flux']*f0_order, yerr=km2aerr*f0_order, label=labelK, **param_dictK)
    if len(wcdaul) != 0:
        ax.errorbar(wcdaul['energy'], wcdaul['flux']*f0_order, yerr=wcdaul['flux']*0.25*f0_order, uplims=1, mfc='none', **param_dictW)
    if len(km2aul) != 0:
        ax.errorbar(km2aul['energy'], km2aul['flux']*f0_order, yerr=km2aul['flux']*0.25*f0_order, uplims=1, mfc='none', **param_dictK)

    ax.set_xscale('log')
    ax.set_xlabel('Energy [ TeV ]')
    ax.set_yscale('log')
    ax.set_ylabel('E$^2$Flux [ TeV cm$^{-2}$s$^{-1}$ ] ')
    ax.legend()

    if functype != 'none':
        npoint = len(data)
        ee = np.logspace(np.log10(data['energy'][0])-0.2, np.log10(data['energy'][npoint-1])+0.2, 100)
        ff = SED_func(ee, pars, epiv, f0_order, functype=functype)
        ax.plot(ee, ff, '-', linewidth=2, color=linecolor)
        fferr = SEDerr_func(ee, pars, parerrs, epiv, functype=functype)
        ax.fill_between(ee, ff-ff*fferr, ff+ff*fferr, alpha=0.1, color=linecolor)

# Read sed data and draw
par1 = [8.64900, 2.87449, 0.09440]
par1err = [0.04790, 0.00437, 0.00171]
data1 = pd.read_csv("../Results/Crab/SED_Mor/Crab_SED.txt", delimiter=' ')

par3 = [6.70776, 3.15538, 0.06858]
par3err = [0.11789, 0.02493, 0.02670]
data3 = pd.read_csv("Test/Crab_KM2A_allarray.txt", delimiter=' ')

fig, axs = plt.subplots(figsize=(7, 5), dpi=180)

# Crab
styledict = {'marker': 'd', 'markersize': 5, 'linestyle': '', 'color': 'blue', 'alpha': 1.0}
styledictK = {'marker': 's', 'markersize': 5, 'linestyle': '', 'color': 'orange', 'alpha': 1.0}
SED_ploter_AsyErr(axs, data1, f0_order=1.e-14, epiv=10, pars=par1, parerrs=par1err, functype='LP', param_dictW=styledict, param_dictK=styledictK, labelW='WCDA', labelK='KM2A (full)', linecolor='gray')

# KM2A all array
styledict = {'marker': 'd', 'markersize': 5, 'linestyle': '', 'color': 'magenta', 'alpha': 1.0}
styledictK = {'marker': 's', 'markersize': 5, 'linestyle': '', 'color': 'cyan', 'alpha': 1.0, 'markerfacecolor': 'none', 'zorder': 3}
SED_ploter_AsyErr(axs, data3, f0_order=1.e-16, epiv=50, pars=par3, parerrs=par3err, functype='LP', param_dictW=styledict, param_dictK=styledictK, labelK='KM2A (all)', linecolor='cyan')

axs.set_xbound(0.2, 3000)
axs.set_ybound(1.e-15, 1.e-10)

fig.savefig('Test/LHAASO_Crab.png')
fig.savefig('Test/LHAASO_Crab.pdf')
