import sklearn as sk
import numpy as np
import scipy as sp
import pylab as pl
import nibabel as nb
import warnings

from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
import nilearn.input_data.multi_nifti_masker as mnm

    
def compute_similarity_matrix(mask_data, subj_data):
    '''
    Compute sparse similarity matrix
    
    Parameters
    ----------
    subj_data: fmri data for a given subject
    mask_data: mask to use on the data


    Returns
    -------
    correlations_matrix: matrix of similarity
    '''
    mask_sh = mask_data.shape
    connectivity = image.grid_to_graph(n_x=mask_sh[0], n_y=mask_sh[1],
                                   n_z=mask_sh[2], mask=mask_data)
    values = []
    R = connectivity.row
    C = connectivity.col
    # Fill the matrix
    for r, c in zip(R, C):
        corr_coeff = sp.stats.pearsonr(subj_data[:,r], subj_data[:,c])[0]
        
        if np.isnan(corr_coeff):
            warnings.warn("NaN correlation present --> replaced by 0")
            corr_coeff=0
        
        values.append(corr_coeff)
        
    values = np.array(values)
    pos = np.where(values<0)
    if pos[0].shape>0:
        warnings.warn("Negative correlations present --> replace them by exp(corr/mean_corr)")
        values[pos] = np.exp(values[pos]/values.mean())
        
    #Make correlation matrix symmetric
    values = .5 * (values + values.T)
    
    Jvox = subj_data.shape[1]
    correlations_matrix = sp.sparse.coo_matrix((values,(R, C)), shape=(Jvox, Jvox))
    
    return correlations_matrix
   
#filenames   
mask_fn='gm_maskfile.nii.gz'
subj_files = ['subject1.nii.gz', 'subject2.nii.gz', 'subject3.nii.gz']

#load the data
mask_data = nb.load(mask_fn).get_data()
mask_sh = mask_data.shape

nifti_masker = mnm.MultiNiftiMasker(mask_fn)
Nm = nifti_masker.fit(subj_files)

Masked_data = Nm.fit_transform(subj_files) #shape N*J (nb_pts_tempo*nbvox)


print 'Individual clusterings'
for isubj, subj_fn in enumerate(subj_files):
    print 'subject nb ', isubj, '----'
    #example on one subject - test
    subj_data =  Masked_data[isubj]

    N=subj_data.shape[0]

    #prepare for Pearson coefficient computation
    subj_data_std = sp.std(subj_data,0)
    # replace 0 with large number to avoid div by zero
    subj_data_std[subj_data_std==0] = 1000000
    subj_data_mean = sp.mean(subj_data,0)
    subj_data=(subj_data-subj_data_mean)/subj_data_std

    #compute correlations matrix
    print 'compute correlation matrix'
    corr_matrix_subj = compute_similarity_matrix(mask_data, subj_data)
    if isubj==0:
        corr_matrices = corr_matrix_subj
    else:
        corr_matrices = corr_matrices+corr_matrix_subj
    
    #spectral clustering
    print 'perform clustering'
    #labels = spectral_clustering(corr_matrix, n_clusters=N)    
  
#Groupe-level spectral clustering
K=[100,150]    

#average similarity matrices
sum_corr_matrices = sum(corr_matrices)
average_matrix = sum_corr_matrices/len(subj_files)

#cluster average matrix
print 'Group-level clustering'
for k in K:
    print 'Nb of lcusters:' , k
    group_labels = spectral_clustering(average_matrix, n_clusters=k) 




