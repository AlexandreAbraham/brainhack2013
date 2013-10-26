from sklearn.base import clone, BaseEstimator
from nilearn.input_data.base_masker import filter_and_mask
from nilearn._utils.cache_mixin import cache
from nilearn.input_data import MultiNiftiMasker
from joblib import Memory, Parallel, delayed
import warnings
import nibabel
from nilearn._utils.class_inspect import get_params


def subject_covariance(
        estimator, niimgs, mask_img, parameters,
        confounds=None,
        ref_memory_level=0,
        memory=Memory(cachedir=None),
        connectivity=None,
        verbose=0,
        copy=True):
    data, affine = cache(
        filter_and_mask, memory=memory, ref_memory_level=ref_memory_level,
        memory_level=2,
        ignore=['verbose', 'memory', 'ref_memory_level', 'copy'])(
            niimgs, mask_img, parameters,
            ref_memory_level=ref_memory_level,
            memory=memory,
            verbose=verbose,
            confounds=confounds,
            copy=copy)
    estimator = clone(estimator)
    if connectivity is not None:
        estimator.fit(data, connectivity=connectivity)
    else:
        estimator.fit(data)
    return estimator.covariance_


class MultiCovariance(BaseEstimator):

    def __init__(self, estimator, smoothing_fwhm=None, mask=None,
                 detrend=None, standardize=None,
                 target_affine=None, target_shape=None,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=0):
        self.estimator = estimator
        self.mask = mask
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.smoothing_fwhm = smoothing_fwhm
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.standardize = standardize
        self.detrend = detrend

    def fit(self, niimgs=None, y=None, confounds=None, connectivity=None):
        """Compute the mask and the components

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the PCA must be calculated. If this is a list,
            the affine is considered the same for all.
        """
        # Hack to support single-subject data:
        if isinstance(niimgs, (basestring, nibabel.Nifti1Image)):
            niimgs = [niimgs]
            # This is a very incomplete hack, as it won't work right for
            # single-subject list of 3D filenames
        # First, learn the mask
        if not isinstance(self.mask, MultiNiftiMasker):
            self.masker_ = MultiNiftiMasker(mask=self.mask,
                                            smoothing_fwhm=self.smoothing_fwhm,
                                            target_affine=self.target_affine,
                                            target_shape=self.target_shape,
                                            low_pass=self.low_pass,
                                            high_pass=self.high_pass,
                                            t_r=self.t_r,
                                            memory=self.memory,
                                            memory_level=self.memory_level)
        else:
            try:
                self.masker_ = clone(self.mask)
            except TypeError as e:
                # Workaround for a joblib bug: in joblib 0.6, a Memory object
                # with cachedir = None cannot be cloned.
                masker_memory = self.mask.memory
                if masker_memory.cachedir is None:
                    self.mask.memory = None
                    self.masker_ = clone(self.mask)
                    self.mask.memory = masker_memory
                    self.masker_.memory = Memory(cachedir=None)
                else:
                    # The error was raised for another reason
                    raise e

            for param_name in ['target_affine', 'target_shape',
                               'smoothing_fwhm', 'low_pass', 'high_pass',
                               't_r', 'memory', 'memory_level']:
                if getattr(self.masker_, param_name) is not None:
                    warnings.warn('Parameter %s of the masker overriden'
                                  % param_name)
                setattr(self.masker_, param_name,
                        getattr(self, param_name))
        if self.masker_.mask is None:
            self.masker_.fit(niimgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        parameters = get_params(MultiNiftiMasker, self)

        # Now compute the covariances

        self.covariances_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(subject_covariance)(
                self.estimator,
                niimg,
                self.masker_.mask_img_,
                parameters,
                memory=self.memory,
                ref_memory_level=self.memory_level,
                confounds=confounds,
                connectivity=connectivity,
                verbose=self.verbose
            )
            for niimg in niimgs)
        return self
