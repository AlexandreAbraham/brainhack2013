from brainhack.datasets import fetch_craddock_2012_test
from nilearn.input_data import NiftiMasker
from joblib import Memory
from sklearn.feature_extraction.image import grid_to_graph
import nibabel
from sklearn.cluster import spectral_clustering
from scipy.sparse import coo_matrix
from brainhack.metrics.cluster import dice
import numpy as np
from os.path import exists
import pylab as pl


dataset = fetch_craddock_2012_test()
masker = NiftiMasker(mask=dataset.mask)
masker.fit()

### Connectivity graph #######################################################

# In order to run spectral clustering, we need a connectivity graph of brain
# voxels. It can be built upon the mask.

mask_data = nibabel.load(dataset.mask).get_data().astype(bool)
connectivity = grid_to_graph(*mask_data.shape, mask=mask_data)

### Covariance estimator #####################################################

# We instantiate the estimator to use on all the subjects.
from brainhack.covariance.pearson import PearsonCorrelation

pearson = PearsonCorrelation(spatial=False)
pearson_spatial = PearsonCorrelation(spatial=True)

### Compute similarity matrices ##############################################
from brainhack.covariance.multi_covariance import MultiCovariance

if not exists('covariances.npy'):

    multi_cov = MultiCovariance(pearson, mask=dataset.mask, standardize=True,
        detrend=True, memory=Memory(cachedir='nilearn_cache'), memory_level=1)
    # Should add low_pass = 0.1 ?

    multi_cov.fit(dataset.func, connectivity=connectivity)
    np.save('covariances.npy', multi_cov.covariances_)

covariances = np.load('covariances.npy')

'''
if not exists('covariances_spatial.npy'):

    multi_cov_spatial = MultiCovariance(pearson_spatial, mask=dataset.mask,
        standardize=True, detrend=True,
        memory=Memory(cachedir='nilearn_cache'), memory_level=1)

    multi_cov_spatial.fit(dataset.func, connectivity=connectivity)
    np.save('covariances_spatial.npy', multi_cov_spatial.covariances_)

covariances_spatial = np.load('covariances_spatial.npy')

'''
### Reproduce figure #1 of Craddock paper ####################################
'''

rt = np.hstack([c.data for c in covariances])
rs = np.hstack([c.data for c in covariances_spatial])

# Split points that have same signs and opposite signs
rr = rt * rs
rt_plus = rt[rr >= 0]
rs_plus = rs[rr >= 0]
rt_minus = rt[rr < 0]
rs_minus = rs[rr < 0]

pl.figure(figsize=(8, 8))

pl.scatter(rt_plus, rs_plus)
pl.scatter(rt_minus, rs_minus, c='r')


pl.xlim(-1., 1.)
pl.ylim(-1., 1.)

pl.savefig('craddock_figure_1.png')
'''

### Helper Function ##########################################################

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

### First group strategy: simply average similarity matrices #################

group_similarity_matrix = np.mean(covariances, axis=0)
row, col = group_similarity_matrix.nonzero()

# Threshold the matrix
data = group_similarity_matrix.data
data[data < .5] = 0.
group_similarity_matrix = coo_matrix((data, (row, col)))

# Run clustering
group_maps = spectral_clustering(group_similarity_matrix,
        n_clusters=50, assign_labels='discretize')
nibabel.save(masker.inverse_transform(group_maps + 1),
             'group_maps_1.nii.gz')

### We try to reproduce the DICE coefficient comparison ######################
n_subs = len(covariances)

cluster_dices = []
cluster_sizes = range(50, 350, 50)  # range(0, 1050, 50):


for n_clusters in cluster_sizes:
    print('n_clusters %d' % n_clusters)
    if not exists('dice_%d.npy' % n_clusters):
        dices = []
        for i in range(n_subs):

            print('%d/%d' % (i + 1, n_subs))

            sub_cov = covariances[i].copy()
            group_similarity_matrix = np.mean(
                    covariances[np.delete(np.arange(n_subs), i)], axis=0)
            row, col = group_similarity_matrix.nonzero()

            # Threshold the matrix
            data = group_similarity_matrix.data
            data[data < .5] = 0.
            group_similarity_matrix = coo_matrix((data, (row, col)))

            sub_data = sub_cov.data
            sub_data[sub_data < .5] = 0.
            sub_matrix = coo_matrix((sub_data, (row, col)))

            # Run clustering
            group_maps = spectral_clustering(group_similarity_matrix,
                    n_clusters=n_clusters, assign_labels='discretize')
            sub_map = spectral_clustering(sub_matrix,
                    n_clusters=n_clusters, assign_labels='discretize')
            dices.append(dice(clustering_to_connectivity(group_maps),
                              clustering_to_connectivity(sub_map)))
        np.save('dice_%d.npy' % n_clusters, dices)
    dices = np.load('dice_%d.npy' % n_clusters)
    cluster_dices.append(dices)

pl.boxplot(cluster_dices, positions=cluster_sizes, widths=30)
pl.xlim(cluster_sizes[0] - 30, cluster_sizes[-1] + 30)
pl.show()
### Second group strategy ####################################################

'''
clusters = []
for i, cov in enumerate(covariances):
    # Threshold covariance
    data = cov.data
    data[data < .5] = 0.
    cov = coo_matrix((data, (cov.row, cov.col)))
    labels = spectral_clustering(cov, n_clusters=50,
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
        n_clusters=50, assign_labels='discretize')
nibabel.save(masker.inverse_transform(group_maps + 1),
             'group_maps_2.nii.gz')
'''
