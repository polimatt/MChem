# Python 3.13.0

from itertools import combinations

import numpy as np # v==2.1.3
import numpy.linalg as la
import pandas as pd # v==2.2.3
import scipy.stats # v==1.14.1
import matplotlib.pyplot as plt # v==3.9.2
from matplotlib.patches import Rectangle

from statsmodels.stats.outliers_influence import variance_inflation_factor # v==0.14.4
from matplotlib_venn import venn3 # v==1.1.1
import networkx as nx # v==3.4.2

# https://sites.google.com/site/mb3gustame/spatial-analysis/principal-coordinates-of-neighbour-matrices


def eucl_dist(coordinates:np.ndarray):
    '''
    
    '''

    dist_matr = np.zeros((len(coordinates),len(coordinates)))
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            dist_matr[i,j] = la.norm(coordinates[i,:]-coordinates[j,:])

    return dist_matr











# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Preprocessing methods

def standardise(matrix:np.ndarray):
    return np.divide(np.subtract(matrix,np.mean(matrix,axis=0)) , np.std(matrix,axis=0))

def SNV(x:np.ndarray)->np.ndarray:
    mean = np.mean(x,axis=1)
    mean = np.reshape(mean,(len(mean),1))
    std = np.std(x,axis=1)
    std = np.reshape(std,(len(std),1))
    return np.divide(np.subtract(x, mean) , std)

def double_centering(D:np.ndarray,mode:str='dissimilarity'):
        
        assert mode in ['dissimilarity','similarity'], 'The mode must either be "dissimilarity" (default) or "similarity"'

        if mode == 'dissimilarity':
            A = -.5 * np.pow(D,2)
        elif mode == 'similarity':
            A = D.copy()

        n = len(D)
        ONE = np.ones_like(D)
        I = np.diag(np.diag(ONE))

        DELTA1 = (I - (ONE@ONE.T)/n) @ A @ (I - (ONE@ONE.T)/n)

        if mode == 'similarity':
            DELTA1 /= np.sqrt(2)

        return DELTA1

def check_VIF(X, threshold:int|float=5):
    # Calculate VIF for each feature

    vif_values = np.array([variance_inflation_factor(X, i) for i in range(np.shape(X)[1])])

    indx_to_keep = np.where(vif_values < threshold)[0]
    filtered_X = X[:,indx_to_keep]

    return vif_values, indx_to_keep, filtered_X

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Other

def mintree_threshold(dist_matrix):
    '''
    
    https://networkx.org/documentation/stable/auto_examples/graph/plot_mst.html
    '''
    edges_list = []
    for i in range(len(dist_matrix[:,0])):
        for j in range(i+1,len(dist_matrix[:,0])):
            edges_list.append((i,j,{'weight': dist_matrix[i,j]}))

    G = nx.Graph()
    G.add_edges_from(edges_list)
    T = nx.minimum_spanning_tree(G)

    edge_dist = []
    for edge in T.edges:
        edge_dist.append(T.get_edge_data(edge[0],edge[1])['weight'])

    return float(np.max(edge_dist))

def get_neighb_matr(array:np.ndarray,threshold:int|float,threshold_mult:str|float=4):
    '''
    Returns the neighbour matrix with the values above the specified threshold being set at threshold_mult * threshold.

    References:\n
    D. Borcard and P. Legendre, Ecol. Model., 2002, 153, 51-68.
    '''
    neighb_array = array.copy()
    neighb_array[np.where(neighb_array > threshold)] = threshold_mult * threshold
    return neighb_array

def intersections3(R2_list:list|np.ndarray):
    A = R2_list[0]
    B = R2_list[1]
    C = R2_list[2]
    AB = R2_list[3]
    AC = R2_list[4]
    BC = R2_list[5]
    ABC = R2_list[6]

    a = ABC - BC
    b = ABC - AC
    c = ABC - AB

    d = ((AC - a - C) + (BC - b - C)) /2
    f = ((BC - c - B) + (AB - a - B)) /2
    e = ((AB - b - A) + (AC - c - A)) /2

    g = ((A - d - f - a) + (B  - e - d - b) + (C  - e - f - c)) /3

    res = 1 - ABC

    return a,b,c,d,e,f,g,res

def varpart_venn3(venn_dict:dict,save_path:str=None,rounding:int=4,title:str=None):
    '''
    Draw a Venn diagram of the intersections between three explanatory variables (X1, X2, X3) for variation partitioning (varpart).

    - venn_dict = {
        \t\t\t            'X1': R2_a_X1,
        \t\t\t             'X2': R2_a_X2,
        \t\t\t             'X3': R2_a_X3,
        \t\t\t             'X1 ∪ X2': R2_a_(X1 ∪ X2),
        \t\t\t             'X1 ∪ X3': R2_a_(X1 ∪ X3),
        \t\t\t             'X2 ∪ X3': R2_a_(X2 ∪ X3),
        \t\t\t             'X1 ∪ X2 ∪ X3': R2_a_(X1 ∪ X2 ∪ X3),
        \t\t\t              }
    '''

    venn_dict_keys = list(venn_dict.keys())
    venn_dict_values = list(venn_dict.values())

    A = venn_dict_values[0]
    B = venn_dict_values[1]
    C = venn_dict_values[2]
    a,b,c,d,e,f,g,res = intersections3(venn_dict_values)
    
    subsets = [a,b,c,d,e,f,g]
    for i in range(len(subsets)):
        subsets[i] = np.round(subsets[i],rounding)

    subsets_str = ['a','b','c','d','e','f','g']
    subsets_abs = tuple(np.abs((a, b, d, c, f, e, g)))

    labels = (f'{venn_dict_keys[0][0]}\n(R$^{{2}}_{{\\mathrm{{a}}}}$ = {np.round(A,rounding)})',
                f'{venn_dict_keys[1][0]}\n(R$^{{2}}_{{\\mathrm{{a}}}}$ = {np.round(B,rounding)})',
                f'{venn_dict_keys[2][0]}\n(R$^{{2}}_{{\\mathrm{{a}}}}$ = {np.round(C,rounding)})')
   
    # depict venn diagram
    v = venn3(subsets=subsets_abs,
              set_labels=labels,
              set_colors=("orange", "blue", "red"), alpha=0.7)
    
    pos = ['100','010','001','110','011','101','111']

    for i in range(len(pos)):
        v.get_label_by_id(pos[i]).set_text(f'[{subsets_str[i]}]\n{subsets[i]}')

    fig = plt.gcf()
    ax = plt.gca()

    rect_coord = (np.min(ax.get_xlim())-.15,np.min(ax.get_ylim())-.2)
    rect = plt.Rectangle(rect_coord,
                         width  = np.max(ax.get_xlim())-np.min(ax.get_xlim()) + 0.15*2,
                         height = np.max(ax.get_ylim())-np.min(ax.get_ylim()) + 0.3,
                         edgecolor='k',ls='-',lw=1,facecolor='none',clip_on=False)
    
    ax.add_patch(rect)

    ax.text(rect_coord[0]+0.01,rect_coord[1]+0.02,f'[h] = {np.round(res,rounding)}')
    
    if title == None: title = "Variation Partitioning Venn Diagram"
    ax.text(np.mean(plt.xlim()),np.max(plt.ylim())+.15,title,fontsize=15,ha='center')

    if save_path != None:
        fig.savefig(save_path, dpi = 600, facecolor = '#fff', bbox_inches='tight')

def varpart_barplot3(venn_dict:dict,save_path:str=None,rounding:int=4,title:str=None):
    '''
    Draw an UpSet plot-style bar plot of the intersections between three explanatory variables (X1, X2, X3) for variation partitioning (varpart).

    - venn_dict = {
        \t\t\t            'X1': R2_a_X1,
        \t\t\t             'X2': R2_a_X2,
        \t\t\t             'X3': R2_a_X3,
        \t\t\t             'X1 ∪ X2': R2_a_(X1 ∪ X2),
        \t\t\t             'X1 ∪ X3': R2_a_(X1 ∪ X3),
        \t\t\t             'X2 ∪ X3': R2_a_(X2 ∪ X3),
        \t\t\t             'X1 ∪ X2 ∪ X3': R2_a_(X1 ∪ X2 ∪ X3),
        \t\t\t              }
    '''
    
    venn_dict_keys = list(venn_dict.keys())
    venn_dict_values = list(venn_dict.values())

    label_list = [venn_dict_keys[0][0],venn_dict_keys[1][0],venn_dict_keys[2][0],
                  '[a]','[b]','[c]','[d]','[e]','[f]','[g]','[h]']
    venn_dict_keys

    a,b,c,d,e,f,g,res = intersections3(venn_dict_values)
    intsxn = [a,b,c,d,e,f,g,res] #a,b,c,d,f,e,g,res
    intsxn = [venn_dict_values[0],venn_dict_values[1],venn_dict_values[2]] + intsxn

    fig, ax = plt.subplots()
    x_ticks = np.arange(len(intsxn))
    barlist = ax.bar(x_ticks,intsxn)

    for i in x_ticks:
        ax.text(i,intsxn[i]+.005,f'{np.round(intsxn[i],rounding)}',ha='center')

    ax.set_xticks(x_ticks,label_list,rotation=45,ha='right')
    ax.set_ylabel('R$^{2}_{\\mathrm{a}}$',fontsize=12)

    if title == None: title = "Variation Partitioning Bar Plot"
    ax.set_title(title,fontsize=15)

    if save_path != None:
        fig.savefig(save_path, dpi = 600, facecolor = '#fff', bbox_inches='tight')


def nodes_vs_edges_matrix(nodes,edges):
    edges_arr = np.array(edges)
    E = []

    # loop through all the previous edges (seems like it's working, at least for relatively unbranched graphs)
    for node in nodes:
        
        row = np.zeros(len(edges))

        newsearch = np.array([node])
        while newsearch.size>0:
            for n in newsearch:
                row[np.where(edges_arr[:,1]==n)] = 1
                newsearch = edges_arr[np.where(edges_arr[:,1]==n)][:,0]

        E.append(row)

    E = np.array(E)

    return E



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Multivariate methods

def PCA(matrix:np.ndarray,n_components:int=None):
    '''
    Calculate the PCA of an observations vs variables matrix.
    '''
    Y = matrix.copy()
    Y_c = np.subtract(Y,np.mean(Y,axis=0))
    # n is number of observations (samples)
    n = np.shape(Y)[0]

    S = (1/(n-1)) * (Y_c.T @ Y_c)
    
    eigvals, U = la.eigh(S) # U = eigvecs

    # negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[np.isclose(eigvals, 0)] = 0

    # eigvals might not be ordered, so we first sort them, then analogously sort the eigenvectors by the ordering of the eigenvalues too
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    U = U[:, idxs_descending]

    proportion_explained = eigvals / np.sum(eigvals)

    # scores F
    F = Y_c @ U
    F[np.isclose(F, 0)] = 0

    max_n_components = len(eigvals) - len(np.where(eigvals==0)[0])

    if n_components == None:
        n_components = max_n_components

    elif n_components > max_n_components:
        print('The max number of PCs is min(np.shape(matrix)[0]-1,np.shape(matrix)[1])\nn_components will be now set at that value')
        n_components = max_n_components

    U = U[:, :n_components]
    eigvals = eigvals[:n_components]
    proportion_explained = proportion_explained[:n_components]
    F = F[:, :n_components]
    
    return eigvals, proportion_explained, U, F


def PCoA(distance_matrix:np.ndarray,variables:np.ndarray=[],number_of_dimensions:int=None,mode:str='dissimilarity'):
    '''
    Returns eigvals, eigvecs, coordinates, proportion_explained.

    mode must be either "dissimilarity" (default) or "similarity".

    References:\n
    lookmanolowo.web.illinois.edu (https://lookmanolowo.web.illinois.edu/2024/03/13/principal-coordinate-analysis-hands-on/, accessed November 2024).\n
    Scikit-bio v. 0.6.2 (https://github.com/scikit-bio/scikit-bio/blob/4cc395627ac7147b3451b585aeffa09efc75057e/skbio/stats/ordination/_principal_coordinate_analysis.py#L25, accessed November 2024).
    '''

    assert np.all(distance_matrix == distance_matrix.T), 'The distance matrix must be symmetric.'

    # square the matrix to emphasize the hidden structure within the dataset
    D = distance_matrix.copy()
    D = D.astype(float)

    W = double_centering(D,mode=mode)

    eigvals, eigvecs = la.eigh(W)

    negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[negative_close_to_zero] = 0

    # eigvals might not be ordered, so we first sort them, then analogously sort the eigenvectors by the ordering of the eigenvalues too
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]

    # Only return positive eigenvalues and eigenvectors
    num_positive = (eigvals > 0).sum()
    eigvecs = eigvecs[:, :num_positive]
    eigvals = eigvals[:num_positive]

    sum_eigenvalues = np.sum(eigvals)

    proportion_explained = eigvals / sum_eigenvalues
 
    # In case eigh is used, eigh computes all eigenvectors and (-)ve values.
    # So if number_of_dimensions was specified, we manually need to ensure only the requested number of dimensions (number of eigenvectors and eigenvalues, respectively) are returned.
    if number_of_dimensions != None and number_of_dimensions<=len(eigvals):
        eigvecs = eigvecs[:, :number_of_dimensions]
        eigvals = eigvals[:number_of_dimensions]
        proportion_explained = proportion_explained[:number_of_dimensions]

    # Scale eigenvalues to have length = sqrt(eigenvalue). This works because la.eigh returns normalized eigenvectors.
    # Each row contains the coordinates of the objects in the space of principal coordinates.
    # Note that at least one eigenvalue is zero because only n-1 axes are needed to represent n points in a euclidean space.
    coordinates = eigvecs * np.sqrt(eigvals)

    if len(variables) > 1:
        # Contrary to principal component analysis, the relationships between the principal coordinates and the original descriptors
        # are not provided by a principal coordinate analysis. Indeed the descriptors, from which distances were initially computed
        # among the objects, do not play any role during the calculation of the PCoA from matrix D.
        # However, computing the projections of descriptors in the space of the principal coordinates to produce biplots is fairly simple (Legendre & Legendre 2012)

        # center the matrix of the original descriptors
        Y_c = variables - np.mean(variables,axis=0)
        U_st = standardise(eigvecs)
        n = np.shape(eigvecs)[0]
        S_pc = (Y_c.T @ U_st) / (n-1)
        U_proj = np.sqrt(n-1) * (S_pc @ np.pow(la.inv(np.diag(eigvals)),.5))

    else: U_proj = []

    return eigvals, eigvecs, U_proj, coordinates, proportion_explained


def OLS(Y:np.ndarray,X:np.ndarray):
    '''
    Ordinary leat squares regression.
    '''
    return X @ la.inv(X.T @ X) @ X.T @ Y

def RDA(Y:np.ndarray,X:np.ndarray,confidence:float=0.95,verbose:bool=True):
    '''
    
    P. Legendre and L. Legendre, in Developments in Environmental Modelling, ed. P. Legendre and L. Legendre, Elsevier, 2012, vol. 24, ch. 11, pp. 625-710.
    '''

    assert np.shape(Y)[0] == np.shape(X)[0], 'Y must have n objects and p variables (n x p X must have n objects and m variables (n x m).'
    n, p = np.shape(Y)
    m = np.shape(X)[1]
    assert m != (n-1), 'm >= (n-1): The system is overdetermined. If m == (n-1), R^2 cannot be calculated'
    if m > n-1: print('m > n-1: the system is overdetermined, X has too many explanatory variables')

    # Center the matrices Y and X
    Y_c = np.subtract(Y,np.mean(Y,axis=0))

    # Do Ordinary least squares (OLS) on X and Y
    Y_hat = OLS(Y_c,X)
    B = la.inv(X.T @ X) @ X.T @ Y_c

    # Y_hat = la.lstsq(Y_c,X)[0]

    # Perform PCA on Y_hat to find the canonical values
    eigvals_can, _, U_can, Z_can = PCA(Y_hat)

    # Calculate the residual values
    Y_res = np.subtract(Y_c,Y_hat)

    # Perform PCA on Y_res to find the non-canonical values
    eigvals_noncan, _, U_noncan, Z_noncan = PCA(Y_res)

    # Calculate the proportion exaplained from the sum of the eigenvalues we have
    all_eigvals_sum = np.sum( np.append(eigvals_can,eigvals_noncan) )
    proportion_explained_can = eigvals_can / all_eigvals_sum
    proportion_explained_noncan = eigvals_noncan / all_eigvals_sum

    # Calculate the overall canonical R^2
    R2_YX = np.divide( np.var(Y_hat), np.var(Y_c) )
    # Calculate the overall adjusted R^2
    R2_a = 1 - (1 - R2_YX) * ( (n-1) / (n-m-1) )

    # Calculate the r for each axis
    F = Y_c @ U_can
    r_k = np.zeros(np.shape(F)[1])
    for i in range(np.shape(F)[1]):
        r_k[i] = scipy.stats.linregress(F[:,i], Z_can[:,i])[2]

    # Get the contribution of the explanatory variables X to the canonical ordination axes
    R_XZ = np.zeros((np.shape(X)[1],np.shape(Z_can)[1]))
    for i in range(np.shape(R_XZ)[0]):
        for j in range(np.shape(R_XZ)[1]):
            R_XZ[i,j] = scipy.stats.linregress(X[:,i], Z_can[:,j])[2]
            
    # Get the matrix of biplot scores in scaling type 1 (BS_1) for the explanatory variables
    BS_1 = np.zeros_like(R_XZ)
    for i in range(np.shape(BS_1)[1]):
        BS_1[:,i] = R_XZ[:,i] * np.sqrt(proportion_explained_can)[i]

    # Run the F-test
    # (1) Get the degrees of freedom
    # (1.a) in the case of standardised values
    if np.all(np.isclose(Y,standardise(Y))): v1 , v2 = (m*p) , (p*(n-m-1))

    # (1.b) in the case of non-standardised values
    else: v1 , v2 = m , (n-m-1)

    # (2) Calculate the F-statistic
    F_statistic = (R2_YX / v1) / ( (1 - R2_YX) / v2)

    # (3) Compare with the F critical value and test the null-hypothesis
    crit_F_value = scipy.stats.f.ppf(q=confidence, dfn=v1, dfd=v2)
    H_0 = F_statistic < crit_F_value
    
    # Calculate the p-value by getting the right-hand side of the distribution
    p_value = 1 - scipy.stats.f.cdf(F_statistic, dfn=v1, dfd=v2)

    # If verbose == True, print some info regaring the F-test
    if verbose:
        print(f'The null-hypothesis is {H_0} in the {confidence*100}% confidence range (F-statistic = {np.round(F_statistic,2)}, critical F value = {np.round(crit_F_value,2)}).')
        print(f'p = {np.round(p_value,3)}')

    return (B,eigvals_can, proportion_explained_can, U_can, Z_can, eigvals_noncan,
            proportion_explained_noncan, U_noncan, Z_noncan, R2_YX, R2_a, r_k,
            R_XZ, BS_1, F_statistic, crit_F_value, p_value)



def dbMEM(distance_matrix:np.ndarray,threshold:float,number_of_dimensions:int=None,threshold_mult:int|float=4):
    '''
    Returns eigvals, eigvecs, coordinates, proportion_explained.

    References:\n
    D. Borcard and P. Legendre, Ecol. Model., 2002, 153, 51-68.
    '''

    if number_of_dimensions == None: number_of_dimensions = np.shape(distance_matrix)[0]

    neighb_array = get_neighb_matr(distance_matrix,threshold,threshold_mult)
    eigvals, eigvecs, _, coordinates, proportion_explained = PCoA(neighb_array,[],number_of_dimensions)

    return eigvals, eigvecs, coordinates, proportion_explained

def varpart(Y:np.ndarray,Xs:list,X_labels:list=[],title:str=None,save_path:str=None,diagram:str='Venn',rounding:int=4):

    combo_list = []
    str_combo_list = []

    if X_labels == []:
        for i in range(len(Xs)):
            X_labels.append(f'X{i+1}')

    for i in range(1,len(Xs)+1):
        combo_list += list(combinations(range(len(Xs)),i))

        str_combo_list += list(combinations(X_labels,i))

    R2_list = []
    for combo in combo_list:
        for i in range(len(combo)):
            idx = combo[i]
            if i == 0:
                X_combo = Xs[idx]
            else:
                X_combo = np.concat((X_combo,Xs[idx]),axis=1)

        R2_list.append(RDA(Y,X_combo,verbose=0)[-7])

    R2_list = np.array(R2_list)

    venn_dict = {}
    for i in range(len(R2_list)):
        venn_dict[str_combo_list[i]] = float(np.round(R2_list[i],4))

    # Plot the intersections
    
    if diagram.upper() == 'VENN':
        if len(Xs) == 2:
            pass #varpart_venn2
        elif len(Xs) == 3:
            varpart_venn3(venn_dict,title=title,save_path=save_path,rounding=rounding)
    
    elif diagram.upper() in ['BAR','UPSET']:
        if len(Xs) == 2:
            pass #intsxn_barplot2
        elif len(Xs) == 3:
            varpart_barplot3(venn_dict,title=title,save_path=save_path,rounding=rounding)

    return R2_list, venn_dict



def AEM(E:np.ndarray=[],nodes:np.ndarray=[],edges:np.ndarray=[],w:np.ndarray|np.matrix=[]):

    if len(E) == 0:
        assert len(edges) != 0, 'If E is not given, please provide at least the edges.'

        if len(nodes) == 0: nodes = np.unique(edges)
        
        E = nodes_vs_edges_matrix(nodes,edges)

    if len(w) == 0:
        w = np.ones_like(E)

    assert np.shape(w) == np.shape(E), 'w must have shape == (len(nodes),len(edges))'

    E_w = E * w
    # E_wc = E_w - np.mean(E_w,axis=0)

    eigvals, proportion_explained, U, F = PCA(E_w) # PCA already centers the matrix

    return E, eigvals, proportion_explained, U, F