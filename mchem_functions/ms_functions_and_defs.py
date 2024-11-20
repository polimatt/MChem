# Python 3.13
# Importing Libraries
import re
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# from corems.transient.input.brukerSolarix import ReadBrukerSolarix # v. 1.6.0
# from corems.encapsulation.factory.parameters import MSParameters
# from corems.mass_spectrum.calc.Calibration import MzDomainCalibration
# from corems.molecular_id.search.molecularFormulaSearch import SearchMolecularFormulas

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Defining Constants
nom_m_dict = {
    'C': 12,
    'H': 1,
    'N': 14,
    'O': 16,
    'P': 31,
    'S': 32,
}

m_dict = {
    'C': 12, #12C
    'H': 1.007825031898, #1H
    'N': 14.00307400425, #14N
    'O': 15.99491461926, #16O
    'P': 30.97376199768, #31P
    'S': 31.97207117354, #32S
}

vk_areas = {
    'Carbohydrates':    {'O/C': [.8,50],     'H/C': [1.65,2.7]                         }, #[[O/C_min,O/C_max],[H/C_min,H/C_max]]
    'Lipids':           {'O/C': [-1,0.6],    'H/C': [1.32,50],    'N/C': [-1,0.126]    },
    'Lignins':          {'O/C': [.21,.44],   'H/C': [.86,1.34]                         },
    'Tannins':          {'O/C': [.16,.84],   'H/C': [.7,1.01]                          },
    'Amino sugars':     {'O/C': [.61,50],    'H/C': [1.45,50],    'N/C': [0.07,0.2]    }, #a third item to indicate that this class contains N (put N/C ratio)
    'Peptides1':        {'O/C': [.12,.6],    'H/C': [.9,2.5],     'N/C': [0.126,0.7]   },
    'Peptides2':        {'O/C': [.6,1],      'H/C': [1.2,2.5],    'N/C': [0.2,0.7]     },

    # 'Peptides':{'O/C': [.17,.48],'H/C': [1.33,1.84],'N/C': [0.126,0.7]},
}

vk_region_colours = {
    'Lipids': '#F5C308', #dark yellow
    'Peptides': '#BD3A53', #purple

    'Carbohydrates': '#0040FF', #blue
    'Amino sugars': '#7C9EBF', #sky blue
    
    'Lignins': '#33251E', #Dark Brown
    'Tannins': '#CE833B', #brown 

    'Unassigned': '#e6e6e6' #light grey
}

ai_boundaries = (.5,.67)
ai_colours = ['tab:blue','tab:orange','k']

normal_fontsize = 12
title_fontsize = 15

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Defining Functions

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## For raw MS data processing


#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## For further processing

def dbe(c:int|np.ndarray,h:int|np.ndarray,n:int|np.ndarray)->float|np.ndarray:
    return c -.5*h + .5*n

def dbe_o(c:int|np.ndarray,h:int|np.ndarray,n:int|np.ndarray,o:int|np.ndarray)->float|np.ndarray:
    return dbe(c,h,n) - o

def roman_to_integer(roman_no:str)->int:
    '''
    Convert a roman numeral to its corresponding integer value.
    '''
    roman_no = roman_no.lower()

    tot = roman_no.count('i') + roman_no.count('v') * 5 + roman_no.count('x') * 10 + roman_no.count('l') * 50 + roman_no.count('c') * 100 + roman_no.count('d') * 500 + roman_no.count('m') * 1000
    if 'iv' in roman_no: tot -= roman_no.count('iv') * 2
    if 'ix' in roman_no: tot -= roman_no.count('ix') * 2
    if 'xl' in roman_no: tot -= roman_no.count('xl') * 20
    if 'xc' in roman_no: tot -= roman_no.count('xc') * 20
    if 'cd' in roman_no: tot -= roman_no.count('cd') * 200
    if 'cm' in roman_no: tot -= roman_no.count('cm') * 200

    return tot

def calc_mass(formula:dict,mode:str)->float:
    if mode == 'nominal':
        masses = nom_m_dict
    elif mode == 'monoisotopic':
        masses = m_dict
    
    m = 0
    for e in formula:
        m += formula[e] * masses[e]
    
    return m

def kendrick_mass(mass)->float:
    return mass * 14 / 14.01565

def KMD(formula:dict,decimals=3):
    nom_mass = calc_mass(formula,mode= 'nominal')
    k_nom_mass = np.ceil(kendrick_mass(nom_mass)) #np.round(kendrick_mass(nom_mass),0)

    exact_mass = calc_mass(formula,mode= 'monoisotopic')
    k_exact_mass = kendrick_mass(exact_mass)

    return np.round((k_nom_mass - k_exact_mass)*1000,decimals=decimals)

def Z_star(formula:dict):
    return np.mod(calc_mass(formula,mode= 'nominal'),14) - 14

def kendrick_analysis(formula:dict):
    m = calc_mass(formula,mode='monoisotopic')
    kmass = kendrick_mass(m)
    kmd = KMD(formula)
    z = Z_star(formula)
    return kmass, kmd, z

def MolecFormulaDict(str:str)->dict:
    str = str.replace(' ','')
    numbers = re.findall(r'[0-9]+', str)
    alphabets = re.findall(r'[a-zA-Z]+', str)

    formula = {}

    for i in range(len(numbers)):
        formula[alphabets[i]] = int(numbers[i])
    
    return formula

def LatexMolecFormula(str:str):
    str = str.replace(' ','')
    numbers = re.findall(r'[0-9]+', str)
    alphabets = re.findall(r'[a-zA-Z]+', str)

    final_str = '$'

    for i in range(len(numbers)):
        final_str += f'\\mathrm{{{alphabets[i]}}}_{{{numbers[i]}}}'
    
    final_str += '$'
    return final_str

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## Plotting

def save_fig(fig:plt.Figure,save_path:str):
    fig.savefig(save_path, dpi=600, facecolor = '#fffa', bbox_inches='tight')

def massspectrum_plot(mz:tuple|list|np.ndarray,intensities:tuple|list|np.ndarray,mode:str='bar',title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,xlim:list=[],ylim:list=[],**kwargs):

    if fig == None and ax == None:
        fig, ax = plt.subplots()

    if mode == 'bar':
        ax.bar(mz,intensities,**kwargs)

    if mode == 'plot':
        # sort values by m/z values
        spectrum = pd.DataFrame({'m/z':mz, 'intensity':intensities}).sort_values('m/z')
        ax.plot(spectrum['m/z'],spectrum['intensity'],**kwargs)  
    
    if title != None: ax.set_title(title,fontsize=title_fontsize)
    ax.set_xlabel('$m$/$z$',fontsize=normal_fontsize)
    ax.set_ylabel('Intensity',fontsize=normal_fontsize)
    
    if xlim == []: xlim = [np.min(mz),np.max(mz)]
    if ylim == []: ylim = 0
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if save_path!=None:
        save_fig(fig,save_path)

    return fig, ax

def dbe_plot(mz:tuple|list|np.ndarray,dbe:tuple|list|np.ndarray,title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,xlim:list=[],ylim:list=[],**kwargs):
    '''
    
    '''
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    
    ax.bar(mz,dbe,**kwargs)
    if title != None: ax.set_title(title,fontsize=title_fontsize)
    ax.set_xlabel('$m$/$z$',fontsize=normal_fontsize)
    ax.set_ylabel('DBE',fontsize=normal_fontsize)

    if xlim == []: xlim = [np.min(mz),np.max(mz)]
    ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if save_path!=None:
        save_fig(fig,save_path)

    return fig, ax

def kendrick_plot(kmass,kmd,z_stars=None,z_wanted:int|float=None,title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,xlim:list=[],ylim:list=[],**kwargs):

    if fig == None and ax == None:
        fig, ax = plt.subplots()
    
    if z_wanted != None:
        assert np.any(z_stars) != None, 'Need a list or an array of Z* values (z_stars=)'
        idx = np.where(z_stars==z_wanted)
        kmass = kmass[idx]
        kmd = kmd[idx]
    
    # can give all z_stars and no z_wanted, in this case a single plot will be generated with all z* values in different colours.
    if np.any(z_stars) != None and z_wanted == None:
        Z_stars_unique = np.unique(z_stars)
        for z in Z_stars_unique:
            idx = np.where(z_stars==z)
            ax.scatter(kmass[idx],kmd[idx],marker='.',label=f'Z* = {int(z)}',**kwargs)
        ax.legend(framealpha=1,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    else:
        ax.scatter(kmass,kmd,marker='.',**kwargs)

    ax.set_xlabel('Kendrick Mass [g mol$^{^-1}$]',fontsize=normal_fontsize)
    ax.set_ylabel('Kendrick Mass Defect [mg mol$^{^-1}$]',fontsize=normal_fontsize)  
    
    if xlim != []: ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if title != None:
        if z_wanted != None:
            title += f' (Z* = {int(z_wanted)})'
        ax.set_title(title,fontsize=title_fontsize)

    if save_path!=None:
        save_fig(fig,save_path)

    return fig, ax

def vk_diagram(df:pd.DataFrame,title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,xlim:list=[],ylim:list=[],**kwargs):
    
    if fig == None and ax == None:
        fig, ax = plt.subplots()
    
    ax.scatter(df['O/C'], df['H/C'],marker='.',**kwargs)
    ax.set_xlabel('O/C',fontsize=normal_fontsize) 
    ax.set_ylabel('H/C',fontsize=normal_fontsize)
    ax.legend(framealpha=1,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=normal_fontsize)
    
    if xlim != []: ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if title != None:
        ax.set_title(title,fontsize=title_fontsize)

    if save_path!=None:
        save_fig(fig,save_path)
    
    return fig, ax

def density_vk_diagram(df:pd.DataFrame,cmap='viridis',title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,cbar:bool=True,xlim:list=[],ylim:list=[],**kwargs):

    # from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density

    if 'O/C' not in df.columns:
        df['O/C'] = df['O'] / df['C']
    if 'H/C' not in df.columns:
        df['H/C'] = df['H'] / df['C']

    x = df['O/C']
    y = df['H/C']

    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    if fig == None and ax == None:
        fig, ax = plt.subplots()
    
    mappable = ax.scatter(x, y, c=z, marker='.',cmap=cmap,**kwargs)

    if cbar:
        cbar = fig.colorbar(mappable)
        cbar.set_label('Kernel Density',fontsize=normal_fontsize)

    ax.set_xlabel('O/C',fontsize=normal_fontsize) 
    ax.set_ylabel('H/C',fontsize=normal_fontsize)

    if xlim != []: ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if title != None:
        ax.set_title(title,fontsize=title_fontsize)

    if save_path!=None:
        save_fig(fig,save_path)

    return fig, ax

def molecclass(df:pd.DataFrame,regions:dict=vk_areas.copy()):
    vk_sorted = {}
    idxs_sorted =[]

    if 'N' in df.columns and 'N/C' not in df.columns:
        df['N/C'] = df['N'].to_numpy() / df['C'].to_numpy()

    if 'H' in df.columns and 'H/C' not in df.columns:
        df['H/C'] = df['H'].to_numpy() / df['C'].to_numpy()

    if 'O' in df.columns and 'O/C' not in df.columns:
        df['O/C'] = df['O'].to_numpy() / df['C'].to_numpy()

    for region in regions:
        OC_min = np.min(regions[region]['O/C'])
        OC_max = np.max(regions[region]['O/C'])
        HC_min = np.min(regions[region]['H/C'])
        HC_max = np.max(regions[region]['H/C'])

        region_keys = list(regions[region].keys())

        if region_keys == ['O/C','H/C']:
            vk_sorted[region] = df[(df['O/C']>OC_min)&(df['O/C']<OC_max)&(df['H/C']>HC_min)&(df['H/C']<HC_max)]
            
        elif 'N/C' in region_keys:
            NC_min = np.min(regions[region]['N/C'])
            NC_max = np.max(regions[region]['N/C'])
            vk_sorted[region] = df[(df['O/C']>OC_min)&(df['O/C']<OC_max)&(df['H/C']>HC_min)&(df['H/C']<HC_max)&(df['N/C']>NC_min)&(df['N/C']<NC_max)]
        
        idxs_sorted += list(vk_sorted[region].index)

    if 'Peptides1' in regions.keys() and 'Peptides2' in regions.keys():
        vk_sorted['Peptides'] = pd.concat([vk_sorted['Peptides1'], vk_sorted['Peptides2']]).drop_duplicates()
        vk_sorted.pop('Peptides1',None)
        vk_sorted.pop('Peptides2',None)

    vk_sorted['Unassigned'] = df[~df.index.isin(idxs_sorted)] 

    return vk_sorted

def vk_molecclass(df:pd.DataFrame,regions:dict=vk_areas.copy(),region_colours:dict=vk_region_colours,title:str=None,save_path:str=None,fig:plt.Figure=None,ax:plt.Axes=None,xlim:list=[],ylim:list=[],**kwargs):

    # assert len(region_colours) == len(regions)+1, 'The length of region_colours should be that of the regions + 1 to account for the class "Unassigned".'

    vk_sorted = molecclass(df)

    if 'Peptides1' in regions.keys() and 'Peptides2' in regions.keys():
        # regions.pop('Peptides1',None)
        # regions.pop('Peptides2',None)
        regions['Peptides'] = []

    if fig == None and ax == None:
        fig, ax = plt.subplots()
        
    # zorder determined by how many compounds are present in that class: the more, the lower the zorder
    lenghts = []
    for region in vk_sorted:
        lenghts.append(len(vk_sorted[region]))
    lenghts.sort()
    zorder = {}
    for region in vk_sorted:
        zorder[region] = -lenghts.index(len(vk_sorted[region]))

    # plot the VKD
    for region in vk_region_colours:
        compounds = vk_sorted[region]
        ax.scatter(compounds['O/C'], compounds['H/C'],marker='.',label=region,c=region_colours[region],zorder=zorder[region],**kwargs)

    ax.set_xlabel('O/C',fontsize=normal_fontsize) 
    ax.set_ylabel('H/C',fontsize=normal_fontsize)
    ax.legend(framealpha=1,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=normal_fontsize)
    
    if xlim != []: ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if title != None:
        ax.set_title(title,fontsize=title_fontsize)

    if save_path != None:
        save_fig(fig,save_path)
    
    return fig, ax

def AI(c:int|np.ndarray,h:int|np.ndarray,n:int|np.ndarray,o:int|np.ndarray,s:int|np.ndarray,p:int|np.ndarray)->float|np.ndarray:
    ai = None
    if np.all([isinstance(i, np.ndarray) for i in [c,h,n,o,s,p]]):
        ai = np.zeros(len(c))
        for i in range(len(ai)):
            if (c[i] - o[i] - n[i] - s[i] - p[i]) != 0:
                ai[i] = (1 + c[i] - o[i] - s[i] - .5*h[i]) / (c[i] - o[i] - n[i] - s[i] - p[i])
            else:
                ai[i] = 0

    elif np.all([isinstance(i, int) for i in [c,h,n,o,s,p]]):
        if (c - o - n - s - p) != 0:
            ai = (1 + c - o - s - .5*h) / (c - o - n - s - p)
        else:
            ai = 0

    return ai

def vk_ai(df:pd.DataFrame,ai,values=ai_boundaries,colour=ai_colours,title=None,save_path=None,fig:plt.Figure=None,ax:plt.Axes=None,alpha=.5,xlim:list=[],ylim:list=[],**kwargs):

    values = np.sort(values)
    labels = [f'AI $\\leq$ {values[0]}']

    idxs = []

    idxs.append(np.where(ai<=values[0])[0])

    for i in range(len(values)-1):
        if values[i] != values[-2]:
            labels += [f'{values[i]} < AI $\\leq$ {values[i+1]}']
            idxs.append(np.where((ai>values[i])&(ai<=values[i+1]))[0])
        else:
            labels += [f'{values[i]} < AI < {values[i+1]}']
            idxs.append(np.where((ai>values[i])&(ai<values[i+1]))[0])

    labels += [f'AI $\\geq$ {values[-1]}']
    idxs.append(np.where(ai>=values[-1])[0])

    if fig == None and ax == None:
        fig, ax = plt.subplots()
    
    for i in range(len(idxs)):
        idx = idxs[i]
        ax.scatter(df['O/C'].to_numpy()[idx], df['H/C'].to_numpy()[idx],marker='.',c=colour[i],alpha=alpha,label=labels[i],**kwargs)

    ax.set_xlabel('O/C',fontsize=normal_fontsize) 
    ax.set_ylabel('H/C',fontsize=normal_fontsize)
    ax.legend(framealpha=1,bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,fontsize=normal_fontsize)   
    

    if xlim != []:
        ax.set_xlim(xlim)
    if ylim != []: ax.set_ylim(ylim)

    if title != None:
        ax.set_title(title,fontsize=title_fontsize)

    if save_path!=None:
        save_fig(fig,save_path)

    return fig, ax