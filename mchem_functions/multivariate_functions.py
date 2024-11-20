import numpy as np
import numpy.linalg as la
import networkx as nx

'''
https://sites.google.com/site/mb3gustame/spatial-analysis/principal-coordinates-of-neighbour-matrices

'''

def eucl_dist(coordinates:np.ndarray):

    dist_matr = np.zeros((len(coordinates),len(coordinates)))
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            dist_matr[i,j] = la.norm(coordinates[i,:]-coordinates[j,:])

    return dist_matr

def PCoA(distance_matrix:np.ndarray,number_of_dimensions:int=None):
    '''
    Returns eigvals, eigvecs, coordinates, proportion_explained.

    References:\n
    lookmanolowo.web.illinois.edu (https://lookmanolowo.web.illinois.edu/2024/03/13/principal-coordinate-analysis-hands-on/, accessed November 2024).\n
    Scikit-bio v. 0.6.2 (https://github.com/scikit-bio/scikit-bio/blob/4cc395627ac7147b3451b585aeffa09efc75057e/skbio/stats/ordination/_principal_coordinate_analysis.py#L25, accessed November 2024).
    '''

    assert np.all(distance_matrix == distance_matrix.T), 'The distance matrix must be symmetric.'

    # square the matrix to emphasize the hidden structure within the dataset
    D = distance_matrix.copy()
    D = D.astype(float)

    def double_centering(D:np.ndarray):
        
        D2 = np.pow(D,2)

        n = len(D)

        # get centering matrix A
        J = np.ones_like(D)
        I = np.diag(np.diag(J))
        A = I - (J / n)

        # get W, which encodes the variance and structure of the original dataset as represented by the squared distances, after being adjusted by double-centering
        W = -.5 * la.multi_dot([A, D2, A.T])
        return W
    
    W = double_centering(D)

    eigvals, eigvecs = la.eigh(W)

    negative_close_to_zero = np.isclose(eigvals, 0)
    eigvals[negative_close_to_zero] = 0

    # eigvals might not be ordered, so we first sort them, then analogously sort the eigenvectors by the ordering of the eigenvalues too
    idxs_descending = eigvals.argsort()[::-1]
    eigvals = eigvals[idxs_descending]
    eigvecs = eigvecs[:, idxs_descending]

    # If we return only the coordinates that make sense (i.e., that have a corresponding positive eigenvalue), then Jackknifed Beta Diversity
    # won't work as it expects all the OrdinationResults to have the same number of coordinates.
    # In order to solve this issue, we return the coordinates that have a negative eigenvalue as 0
    num_positive = (eigvals >= 0).sum()
    eigvecs[:, num_positive:] = np.zeros(eigvecs[:, num_positive:].shape)
    eigvals[num_positive:] = np.zeros(eigvals[num_positive:].shape)

    sum_eigenvalues = np.sum(eigvals)

    proportion_explained = eigvals / sum_eigenvalues
 
    # In case eigh is used, eigh computes all eigenvectors and (-)ve values.
    # So if number_of_dimensions was specified, we manually need to ensure only the requested number of dimensions (number of eigenvectors and eigenvalues, respectively) are returned.
    if number_of_dimensions != None:
        eigvecs = eigvecs[:, :number_of_dimensions]
        eigvals = eigvals[:number_of_dimensions]
        proportion_explained = proportion_explained[:number_of_dimensions]

    # Scale eigenvalues to have length = sqrt(eigenvalue). This works because la.eigh returns normalized eigenvectors.
    # Each row contains the coordinates of the objects in the space of principal coordinates.
    # Note that at least one eigenvalue is zero because only n-1 axes are needed to represent n points in a euclidean space.
    coordinates = eigvecs * np.sqrt(eigvals)

    return eigvals, eigvecs, coordinates, proportion_explained

def get_threshold(dist_matrix):
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

def PCNM(distance_matrix:np.ndarray,threshold:float,number_of_dimensions:int=None,threshold_mult:int|float=4):
    '''
    Returns eigvals, eigvecs, coordinates, proportion_explained.

    References:\n
    D. Borcard and P. Legendre, Ecol. Model., 2002, 153, 51-68.
    '''

    neighb_array = get_neighb_matr(distance_matrix,threshold,threshold_mult)
    eigvals, eigvecs, coordinates, proportion_explained = PCoA(neighb_array,number_of_dimensions)

    return eigvals, eigvecs, coordinates, proportion_explained