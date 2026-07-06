import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import yaml


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


def find_src_by_name(yaml_file, target_name):
    """
    在 YAML 文件中查找 Name 为指定值的 Src 节点

    Args:
        yaml_file (str): YAML 文件路径
        target_name (str): 要查找的 Name 值（如 "Crab"）

    Returns:
        dict: 匹配的 Src 节点数据（如 Src0/Src1 的内容），未找到返回 None
    """
    with open(yaml_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

        # 遍历 SRC 下的所有键
        if 'SRC' in data:
            for key in data['SRC']:
                if key.startswith('Src') and isinstance(data['SRC'][key], dict):
                    if data['SRC'][key].get('Name') == target_name:
                        return data['SRC'][key]

        if 'DGE' in data:
            for key in data['DGE']:
                if key.startswith('Template') and isinstance(data['DGE'][key], dict):
                    if data['DGE'][key].get('Name') == target_name:
                        return data['DGE'][key]

    return None


def ReadSEDmodelconfig(sedtype, fconfig='../src/Src_SEDModel.yaml'):
    with open(fconfig, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
        for value in data['Tag']:
            if value == sedtype:
                return data[sedtype]
    return None


# Read sed data and draw
# 初始化解析器
parser = argparse.ArgumentParser(description="plot sed of target source:")
parser.add_argument("--DirRes", type=str, help="directory of results")
parser.add_argument("--ParRes", type=str, help="ParRes.yaml")
parser.add_argument("--srcname", type=str, help="name of target source")

# 解析参数
args = parser.parse_args()
dirres = args.DirRes
parres = args.ParRes
srcname = args.srcname
print(f" >>> analysis : {dirres}, target source name: {srcname}")

# 读取目标源的结果
sedpar = parres
srcdata = find_src_by_name(sedpar, srcname)
if srcdata is None:
    print(f" Error: Source {srcname} Not Found")
    exit()
sedpoint = dirres+'/SED_Mor/'+srcname+'_SED.txt'
data = pd.read_csv(sedpoint, delimiter=' ')

# 解析目标源的结果
epiv = srcdata['Epiv']
sedtype = srcdata['SEDModel']['type']
f0_order = srcdata['SEDModel']['F0'][4]
sedmodel = ReadSEDmodelconfig(sedtype)
par = []
parerr = []
for ipar in range(sedmodel['Npar']):
    par.append(srcdata['SEDModel'][sedmodel['Parname'][ipar]][0])
    if ipar == 0:
        parerr.append(srcdata['SEDModel'][sedmodel['Parname'][ipar]][5])
    else:
        parerr.append(srcdata['SEDModel'][sedmodel['Parname'][ipar]][4])


# 绘图
fig, axs = plt.subplots(figsize=(7, 5), dpi=180)

styledict = {'marker': 'd', 'markersize': 5, 'linestyle': '', 'color': 'blue', 'alpha': 1.0}
styledictK = {'marker': 's', 'markersize': 5, 'linestyle': '', 'color': 'orange', 'alpha': 1.0}
SED_ploter_AsyErr(axs, data, f0_order=f0_order, epiv=epiv, pars=par, parerrs=parerr, functype=sedtype, param_dictW=styledict, param_dictK=styledictK, labelW='WCDA', labelK='KM2A', linecolor='gray')

validindex = np.where((data['flux']!=0) & (data['ferrU']!=0))
index = ((data['flux']!=0) & (data['ferrU']!=0))
if len(validindex[0]) != 0:
    if validindex[0][0] != 0:
        index[validindex[0][0]-1] = True
    if validindex[0][len(validindex[0])-1] != len(data)-1:
        index[validindex[0][len(validindex[0])-1]+1] = True
    minenergy = np.min(data[index]['energy'])
    maxenergy = np.max(data[index]['energy'])
    axs.set_xbound(minenergy*0.3, maxenergy*3)
    minflux = np.min(data[index]['flux'])
    maxflux = np.max(data[index]['flux'])
    axs.set_ybound(minflux*0.1*f0_order, maxflux*5*f0_order)

    fig.savefig(dirres+'/SED_Mor/'+srcname+'_SED.png')
    fig.savefig(dirres+'/SED_Mor/'+srcname+'_SED.pdf')
else:
    print(f" Error: Source {srcname} has no flux points")
    exit
