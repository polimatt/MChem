import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import scipy.stats

def SNV(x:np.ndarray)->np.ndarray:
    new = np.zeros_like(x)
    new = ( x - np.mean(x) ) / np.std(x)

    return new

def SNV_matrix(x:np.ndarray)->np.ndarray:
    
    new = np.zeros_like(x)
    for row_index in range(int(np.shape(x)[0])):
        new[row_index,:] = SNV(x[row_index,:])

    return new

def get_pca_data(data:np.ndarray,method:str,no_of_components:int)->tuple:
    
    if method == 'SNV':
        adj_data = SNV_matrix(data)
    
    else: adj_data = data

    pca = decomposition.PCA(n_components=no_of_components)

    scores_values = pca.fit_transform(adj_data)

    variance_ratio = pca.explained_variance_ratio_
    PCs = np.arange(pca.n_components_) + 1

    loadings = np.array(pca.components_)

    return scores_values, variance_ratio, PCs, loadings

def save_figure(fig:plt.Figure,save_path:str):
    fig.savefig(save_path, dpi = 600, facecolor = '#fff', bbox_inches='tight')

def scores_plot(scores_values:np.ndarray,PCs:list|np.ndarray,fig:plt.Figure=None,ax:plt.Axes=None,lines:bool=True,**kwargs):
    '''
    Plot a Scores plot.

    kwargs:
        - title: the title of the plot (preset: 'Scores Plot').
        - c: colour of the data point (can be a list or np.ndrray in case you want to assign colour by a property).
        - label: label of the data points.
        - cmap: the colour map to be used.
        - cbar_title: title of the colour bar.
        - save_path: the directory path where you want to save the plot.
        - titlesize: set the font size of the axes title.
    '''
    title = kwargs.get('title',None)
    c = kwargs.get('c',None)
    cmap = kwargs.get('cmap',None)
    cbar_title = kwargs.get('cbar_title',None)
    save_path = kwargs.get('save_path',None)
    label = kwargs.get('label',None)
    marker = kwargs.get('marker',None)
    variance_ratio = kwargs.get('variance_ratio',[])
    norm = kwargs.get('norm',None)
    cbar_yn = kwargs.get('cbar_yn',True)
    titlesize = kwargs.get('titlesize',12)

    # fig = kwargs.get('fig',None)
    # ax = kwargs.get('ax',None)

    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()

    # ax.scatter(scores_values[:,PCs[0]-1],scores_values[:,PCs[1]-1], c=c, cmap=cmap,label=label,marker=marker)

    mappable = ax.scatter(scores_values[:,PCs[0]-1],scores_values[:,PCs[1]-1], c=c, cmap=cmap,label=label,marker=marker,norm=norm)

    if label != None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    if len(variance_ratio)>0:
        var_text_x = f' ({round(variance_ratio[PCs[0]-1]*100,2)}%)'
        var_text_y = f' ({round(variance_ratio[PCs[1]-1]*100,2)}%)'
    else:
        var_text_x = ''
        var_text_y = ''

    ax.set_xlabel(f'PC{PCs[0]}{var_text_x}')
    ax.set_ylabel(f'PC{PCs[1]}{var_text_y}')

    if lines == True:
        ax.axhline(y=0, color = '#000', linewidth = 1)
        ax.axvline(x=0, color = '#000', linewidth = 1)

    if title == None:
        title = 'Scores Plot'
    ax.set_title(title,fontsize=titlesize)
   
    if cmap != None and cbar_yn == True:
        cbar = fig.colorbar(mappable,orientation="horizontal",shrink=0.75)
    
    if cbar_title != None:
        cbar.set_label(cbar_title)

    if save_path != None:
        save_figure(fig,save_path)

    return mappable

def Hotelling(scores_values:np.ndarray,PCs:list|np.ndarray,ax:plt.Axes,confidence:float=.95,legend:bool=True,color:str='grey',linestyle:str=':',**kwargs):
    '''
    Draw a Hotelling T2 ellipse (95% confidence, though the confidence range can be modified).
    '''
    # 95% Hotelling ellipse
    # from https://stackoverflow.com/questions/46732075/python-pca-plot-using-hotellings-t2-for-a-confidence-interval

    theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
    circle = np.array((np.cos(theta), np.sin(theta)))
    sigma = np.cov(np.array((scores_values[:,PCs[0]-1], scores_values[:,PCs[1]-1])))
    ed = np.sqrt(scipy.stats.chi2.ppf(confidence, 2))
    ell = np.transpose(circle).dot(np.linalg.cholesky(sigma) * ed)
    a, b = np.max(ell[: ,0]), np.max(ell[: ,1]) #95% ellipse bounds
    t = np.linspace(0, 2 * np.pi, 100)
    
    ax.plot(a * np.cos(t), b * np.sin(t), color = color, linestyle = linestyle, label=f'Hotelling T$^{2}$ ({int(confidence*100)}%)',**kwargs)
    if legend == True:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #ax.grid(color = 'lightgray', linestyle = '--')

def scree_plot(PCs:np.ndarray|list,variance_ratio:np.ndarray|list,fig:plt.Figure=None,ax:plt.Axes=None,**kwargs):
    '''
    This function plots Scree plots given a list of principal compontents (PCs) and their associated explained variance (eigenvalue). 

    kwargs:
        - title: the title of the plot (preset: 'Scree Plot').
        - line_colour: the colour of the line (preset: 'darkorange').
        - bar_colour: the colour of the bars of the cumulative sum (preset: 'green').
        - save_path: the directory path where you want to save the plot.
        - titlesize: set the font size of the axis title.
    '''
    
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()

    line_colour = kwargs.get('line_colour',None)
    bar_colour = kwargs.get('bar_colour',None)
    title = kwargs.get('title','Scree Plot')
    save_path = kwargs.get('save_path',None)
    titlesize = kwargs.get('titlesize',None)

    if line_colour == None: line_colour = 'darkorange'
    if bar_colour == None: bar_colour = 'green'

    ax.plot(PCs,variance_ratio*100,'o-',linewidth=2,color=line_colour)

    cumulative_sum = 0
    for i in np.arange(0,5):
        cumulative_sum += variance_ratio[i]*100
        if i == 0:
            ax.bar(PCs[i],cumulative_sum,color=bar_colour,width = 0.5,label='Cum')
        else:
            ax.bar(PCs[i],cumulative_sum,color=bar_colour,width = 0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')

    
    ax.set_title(title,fontsize=titlesize)

    if save_path != None:
        save_figure(fig,save_path)

def loadings_plot(variables:np.ndarray|list,loadings:np.ndarray,PCs:np.ndarray|list,fig:plt.Figure=None,ax:plt.Axes=[],**kwargs):
    '''
    Plot a Loadings plot given a set of variables (numeric or strings) and the principal components (PCs) associated with them.
    This function supports multiple loadings plots in a single figure and single-axes loadings plots.

    kwargs:
        - title: the title of the plot (preset: 'Loadings Plot').
        - c: the colour of the line.
        - save_path: the directory path where you want to save the plot.
        - xlabel: label the x axis (for numerical variables).
        - titlesize: set the font size of the figure title.
        - invert_axis: set whether the x-axis should go from smallest to largest value (False) or from largest to smallest value (True).
    '''

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.subplots(len(PCs),sharex=True)
    
    c = kwargs.get('c',None)
    title = kwargs.get('title','Loadings Plots')
    save_path = kwargs.get('save_path',None)
    titlesize = kwargs.get('titlesize',None)
    invert_axis = kwargs.get('invert_axis',False)

    def just_the_loadings_plots(ax,i):
        ax.axhline(y=0, color = '#000', linewidth = 0.7)#, linewidth = 1, linestyle='--')
        ax.plot(variables,loadings[PCs[i]-1,:],c=c),#linewidth = 2,color = '#008000'
        ax.set_ylabel(f'PC{PCs[i]} Loadings')

    if type(variables[0]) == str:
        locationsx = np.arange(len(variables))
        if len(PCs) == 1:
            just_the_loadings_plots(ax,0)
            ax.set_xticks(locationsx,variables,rotation = 60)
        else:
            for i in range(len(PCs)):
                just_the_loadings_plots(ax[i],i)
                ax[i].set_xticks(locationsx,variables,rotation = 60)

    else:
        if len(PCs) == 1:
            just_the_loadings_plots(ax,0)
            ax0 = ax
            ax_minus1 = ax
        else:
            for i in range(len(PCs)): just_the_loadings_plots(ax[i],i)
            ax0 = ax[0]
            ax_minus1 = ax[-1]

        if invert_axis == False:
            ax0.set_xlim(np.min(variables),np.max(variables))
        else: ax0.set_xlim(np.max(variables),np.min(variables))
        xlabel = kwargs.get('xlabel',None)
        ax_minus1.set_xlabel(xlabel)
     
    if len(PCs) == 1: ax.set_title(title,fontsize=titlesize)
    else: fig.suptitle(title,fontsize=titlesize)

    if save_path != None:
        save_figure(fig,save_path)


def corr_matrix(variables:np.ndarray|list,loadings:np.ndarray,PCs:np.ndarray|list,fig:plt.Figure=None,ax:plt.Axes=None,**kwargs)->np.ndarray:    
    '''Correlation'''
    
    title = kwargs.get('title',f'Correlation Matrix (PC{PCs[0]} vs PC{PCs[1]})')
    ang_labels = kwargs.get('ang_labels',False)
    save_path = kwargs.get('save_path',None)
    cmap = kwargs.get('cmap','summer')

    angle_arr = np.zeros((len(variables),len(variables)))

    angles = []

    for i in range(len(variables)):
        tan_angle = loadings[i,PCs[0]-1] / loadings[i,PCs[1]-1]
        angle = np.arctan(tan_angle)
        if loadings[i,PCs[0]-1] > 0 and loadings[i,PCs[1]-1] < 0 :
            angle = angle + np.pi
        if loadings[i,PCs[0]-1] < 0 and loadings[i,PCs[1]-1] < 0 :
            angle = angle - np.pi

        angles.append(angle)

    for i in range(len(variables)):
        for j in range(len(loadings[:,0])):
            angle_diff = np.abs(angles[i] - angles[j])
            if angle_diff > np.pi:
                angle_diff = (2*np.pi) - angle_diff
            
            angle_arr[i,j] = np.rad2deg(angle_diff)

    z = np.tril(angle_arr, k=0)
    z_cropped = z[1:13,0:12]
    z_zeroless = np.ma.masked_where(z_cropped == 0, z_cropped)

    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color = '#fff', alpha = 1.)
    c = ax.pcolormesh(z_zeroless, cmap=cmap)

    ax.set_title(title)

    if type(variables[0]) == str: text_rot = 60
    else: text_rot = 0

    locationsx = np.arange(len(variables)-1)+0.5
    locationsy = np.arange(len(variables)-1)+0.5
    ax.set_xticks(locationsx, variables[:-1], rotation = text_rot)
    ax.set_yticks(locationsy, variables[1:])

    if ang_labels == True:
        for y in range(z_zeroless.shape[0]):
            for x in range(z_zeroless.shape[1]):
                if z_zeroless[y, x] != np.nan:
                    ax.text(x + 0.5, y + 0.5, '%.0f' % z_zeroless[y, x],
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
       
    ax.set_ylim(len(variables)-1,0)
    cbar = fig.colorbar(c,ticks=range(0,200,20))
    cbar.set_label('Degrees ($\\degree$)')

    if save_path != None:
        save_figure(fig,save_path)

    return angle_arr