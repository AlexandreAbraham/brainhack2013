from brainhack.datasets import fetch_craddock_2012_test
from joblib import Memory
from sklearn.feature_extraction.image import grid_to_graph
import nibabel
from sklearn.cluster import spectral_clustering
from scipy.sparse import coo_matrix
import numpy as np


dataset = fetch_craddock_2012_test()

### Connectivity graph #######################################################

# In order to run spectral clustering, we need a connectivity graph of brain
# voxels. It can be built upon the mask.

mask_data = nibabel.load(dataset.mask).get_data().astype(bool)
connectivity = grid_to_graph(*mask_data.shape, mask=mask_data)

### Covariance estimator #####################################################

# We instantiate the estimator to use on all the subjects.
from brainhack.covariance.pearson import PearsonCorrelation

pearson = PearsonCorrelation(spatial=False)

### Compute similarity matrices ##############################################
from brainhack.covariance.multi_covariance import MultiCovariance

multi_cov = MultiCovariance(pearson, mask=dataset.mask, standardize=True,
        detrend=True, memory=Memory(cachedir='nilearn_cache'), memory_level=1)
# Should add low_pass = 0.1 ?

multi_cov.fit(dataset.func, connectivity=connectivity)

### First group strategy: simply average similarity matrices #################
'''
group_similarity_matrix = np.mean(multi_cov.covariances_, axis=0)
row, col = group_similarity_matrix.nonzero()

# Threshold the matrix
data = group_similarity_matrix.data
data[data < .5] = 0.
group_similarity_matrix = coo_matrix((data, (row, col)))

# Run clustering
group_maps = spectral_clustering(group_similarity_matrix,
        n_clusters=100, assign_labels='discretize')
nibabel.save(multi_cov.masker_.inverse_transform(group_maps + 1),
             'group_maps_1.nii.gz')
'''
### Second group strategy ####################################################


def clustering_to_connectivity(labels):
    r = []
    c = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        for i in indices:
            for j in indices:
                r.append(i)
                c.append(j)
    r = np.hstack(r)
    c = np.hstack(c)
    return coo_matrix((np.ones(len(r)), (r, c)))

clusters = []
for i, cov in enumerate(multi_cov.covariances_):
    # Threshold covariance
    data = cov.data
    data[data < .5] = 0.
    cov = coo_matrix((data, (cov.row, cov.col)))
    labels = spectral_clustering(cov, n_clusters=100,
                                 assign_labels='discretize')
    clusters.append(clustering_to_connectivity(labels))

group_similarity_matrix = np.mean(clusters, axis=0)
row, col = group_similarity_matrix.nonzero()

# Threshold the matrix
data = group_similarity_matrix.data
data[data < .5] = 0.
group_similarity_matrix = coo_matrix((data, (row, col)))

# Run clustering
group_maps = spectral_clustering(group_similarity_matrix,
        n_clusters=100, assign_labels='discretize')
nibabel.save(multi_cov.masker_.inverse_transform(group_maps + 1),
             'group_maps_2.nii.gz')
