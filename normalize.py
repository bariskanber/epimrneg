#!/usr/bin/env python3

"""Epilepsy MRI-negative/positive lesion detection pipeline, Copyright 2020, University College London"""

import numpy as np

def normalize(filename,img):
    if np.sum(np.isinf(img))>0:
        img[np.isinf(img)]=np.nan
    if np.sum(np.isfinite(img))>0:
        print('Normalising volume %s of shape %s...'%(filename,str(img.shape)))
        print('Pre-normalisation mean',np.nanmean(img))
        print('Pre-normalisation std',np.nanstd(img))
        img=(img-np.nanmean(img))/np.nanstd(img)
        print('Post-normalisation mean',np.nanmean(img))
        print('Post-normalisation std',np.nanstd(img))
    return img

