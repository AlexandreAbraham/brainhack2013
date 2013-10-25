from brainhack.datasets import fetch_craddock_2012_test
from nilearn.input_data import MultiNiftiMasker
from joblib import Memory


dataset = fetch_craddock_2012_test()

# Preprocess data
masker = MultiNiftiMasker(mask=dataset.mask, standardize=True, detrend=True,
        memory=Memory(cachedir='nilearn_cache'), memory_level=1)
# Should add low_pass = 0.1 ?

masker.fit()
Xs = masker.transform(dataset.func)
