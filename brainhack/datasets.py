import os
from nilearn.datasets import _get_dataset, _fetch_dataset
from sklearn.datasets.base import Bunch


def fetch_craddock_2012_test(n_subjects=None, data_dir=None, resume=True,
                             verbose=0):
    """Download and load example data from Craddock 2012 work.

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given, all the
        3 subjects are used.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    resume: boolean, optional
        Indicate if dataset fetching can be resumed from previous attempt.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
        - 'func': string list. Paths to functional images
        - 'mask': string. Path to nifti mask file.

    References
    ----------
    `A whole brain fMRI atlas generated via spatially constrained spectral
    clustering <http://www.ncbi.nlm.nih.gov/pubmed/21769991>`_
    Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P.
    & Mayberg, H. S. , Human Brain Mapping, 2012, 33,
    1914-1928 doi: 10.1002/hbm.21333.

    Notes
    -----
    Cameron Craddock provides his code for this work:
    https://github.com/ccraddock/cluster_roi
    """

    # Dataset files
    file_names = ['gm_maskfile.nii.gz', 'subject1.nii.gz', 'subject2.nii.gz',
                  'subject3.nii.gz']
    file_names = [os.path.join('pyClusterROI', fn) for fn in file_names]

    # load the dataset
    try:
        # Try to load the dataset
        files = _get_dataset("craddock_2012_test", file_names,
                             data_dir=data_dir)

    except IOError:
        # If the dataset does not exists, we download it
        url = 'ftp://www.nitrc.org/home/groups/cluster_roi/htdocs/pyClusterROI/pyClusterROI_testdata.1.0.tar.gz'
        _fetch_dataset('craddock_2012_test', [url], data_dir=data_dir,
                           resume=resume, verbose=verbose)
        files = _get_dataset('craddock_2012_test', file_names,
                             data_dir=data_dir)

    # return the data
    return Bunch(mask=files[0], func=files[1:n_subjects])
