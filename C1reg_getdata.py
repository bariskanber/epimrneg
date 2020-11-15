#!/usr/bin/env python3

"""Epilepsy MRI-negative/positive lesion detection pipeline, Copyright 2020, University College London"""

import numpy as np
import nibabel as nib
import glob
import os
import xmltodict
import argparse
import time
import sys
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from normalize import normalize

parser = argparse.ArgumentParser()
parser.add_argument('subjectID', type=str, help='ID of the subject whose parcellation data is requested')
args=parser.parse_args()
assert(os.pathsep not in args.subjectID)

print('Will get parcellation data for '+args.subjectID)
time.sleep(5)

print('Parsing xml')
with open('GIFv3.xml','rt') as f:
    xml=xmltodict.parse(f.read())

regs_TIV={}
for k in xml['document']['labels']['item']: # 0 (non-brain outer), 1 (CSF), 2(GM), 3(WM), 4(DGM), 5(Brain Stem and Pons), 6,7,8 (non-brain)

    if 'lesion' in k['name'].lower(): continue
    if 'vessel' in k['name'].lower(): continue

    tissues=k['tissues'].split(',') # e.g. thalamus: 3,4
    for tissue in tissues:
        if int(tissue) in (1,2,3,4,5):
            regs_TIV[int(k['number'])]=k['name']
            break

regs={}
regs_all={}
for k in xml['document']['labels']['item']: # 2(GM), 3(WM), 4(DGM), 5(Brain Stem and Pons)
    regs_all[int(k['number'])]=k['name']

    if 'lesion' in k['name'].lower(): continue
    if 'vessel' in k['name'].lower(): continue

    tissues=k['tissues'].split(',') # e.g. thalamus: 3,4
    for tissue in tissues:
        if int(tissue) in (2,3,4,5):
            regs[int(k['number'])]=k['name']
            break

print('Additional regions in regs_all compared with regs')
for regno in set(regs_all)-set(regs):
    print(regs_all[regno])

mods=('t1','flair','noddi_ficvf','dti_ad','dti_fa','dti_rd','despot_pd','despot_t1','swan')

fields=('meangray','stdgray')

filename=os.path.join('parcellation_data','parcellation_data_'+args.subjectID+'.csv')

with open(filename,'wt') as f:
    f.write('subjectid,reg,reg_name,vol,norm_vol')
    output_format='%s,%d,%s,%f,%f'

    for modi in range(0,len(mods)):
        for fieldj in range(0,len(fields)):
            f.write(',%s_%s'%(mods[modi],fields[fieldj]))
            output_format+=',%f'

    f.write(',t1QSbound_int_meangray,t1QSbound_int_stdgray,t1QSbound_ext_meangray,t1QSbound_ext_stdgray')

    f.write(',lvol\n')
    output_format+=',%f'*(4) # we output lvol seperately

excludelist={
    }
print('excludelist',excludelist)

subjectDIRprefix=os.path.join('data_rigid_t1_space',args.subjectID)+os.path.sep

for file in glob.glob(subjectDIRprefix+args.subjectID+'*_Parcellation.nii.gz'):
    tokens=os.path.basename(file).split('_')
    subjectID=tokens[0]
    subject_type=subjectID[0]

    if subjectID in excludelist:
        print('Excluding %s because it is labelled "%s"'%(subjectID,excludelist[subjectID]))
        continue

    assert(subject_type in ('C','D','N'))

    nibobj_par=nib.load(file)
    par=nibobj_par.get_fdata()

    print(subjectID,par.shape,par.dtype)
    time.sleep(5)

    lm=None
    if subject_type=='D':
        file_lm=subjectDIRprefix+'%s_Lesion.nii.gz'%subjectID
        if not os.path.exists(file_lm):
            print(file_lm+' does not exist ***************')
            sys.exit(1)

        nibobj_lm=nib.load(file_lm)
        lm=nibobj_lm.get_fdata()
        assert(lm.shape==par.shape)

        if not np.array_equal(np.unique(lm),[0,1]):
            print('Lesion mask is not binary for '+file_lm)
            print('Unique values are: ',np.unique(lm))
            sys.exit(1)

    file_t1=subjectDIRprefix+'%s_T1_Bias_Corrected.nii.gz'%subjectID
    t1=normalize(file_t1,nib.load(file_t1).get_fdata())
    assert(t1.shape==par.shape)

    file_flair=subjectDIRprefix+'%s_FLAIR_to_T1.nii.gz'%subjectID
    flair=normalize(file_flair,nib.load(file_flair).get_fdata())
    assert(flair.shape==par.shape)

    file_noddi_ficvf=subjectDIRprefix+'%s_NODDI_ficvf_to_T1.nii.gz'%subjectID
    noddi_ficvf=nib.load(file_noddi_ficvf).get_fdata()
    assert(noddi_ficvf.shape==par.shape)

    file_dti_ad=subjectDIRprefix+'%s_DTI_AD_to_T1.nii.gz'%subjectID
    dti_ad=nib.load(file_dti_ad).get_fdata()
    assert(dti_ad.shape==par.shape)

    file_dti_fa=subjectDIRprefix+'%s_DTI_FA_to_T1.nii.gz'%subjectID
    dti_fa=nib.load(file_dti_fa).get_fdata()
    assert(dti_fa.shape==par.shape)

    file_dti_rd=subjectDIRprefix+'%s_DTI_RD_to_T1.nii.gz'%subjectID
    dti_rd=nib.load(file_dti_rd).get_fdata()
    assert(dti_rd.shape==par.shape)

    file_despot_pd=subjectDIRprefix+'%s_DESPOT_PD_to_T1.nii.gz'%subjectID
    despot_pd=normalize(file_despot_pd,nib.load(file_despot_pd).get_fdata())
    assert(despot_pd.shape==par.shape)

    file_despot_t1=subjectDIRprefix+'%s_DESPOT_T1_to_T1.nii.gz'%subjectID
    despot_t1=normalize(file_despot_t1,nib.load(file_despot_t1).get_fdata())
    assert(despot_t1.shape==par.shape)

    file_swan=subjectDIRprefix+'%s_SWAN_to_T1.nii.gz'%subjectID
    swan=normalize(file_swan,nib.load(file_swan).get_fdata())
    if len(swan.shape)==4:
        swan=np.mean(swan,axis=3)
        print(file_swan+' has 4 dimensions, averaged over the last one')
    assert(swan.shape==par.shape)

    with open(filename,'at') as f:
        sumvol=0
        for regno in range(0,209):
            if regno not in regs_TIV: continue
            parreg=par[par==regno]
            sumvol+=parreg.size

        obj3D=par.copy().astype(np.int8)

        for regno in range(0,209):
            if regno not in regs: continue
            Q=par==regno

            vol=np.sum(Q)

            t1Q=flairQ=noddi_ficvfQ=dti_adQ=dti_faQ=dti_rdQ=despot_pdQ=despot_t1Q=swanQ=[-1,-1]
            t1QSbound_int=[-1,-1]
            t1QSbound_ext=[-1,-1]

            if vol==0:
                raise Exception('vol is zero for %s of %s'%(regs[regno],subjectID))
            
            t1Q=t1[Q]
            flairQ=flair[Q]
            noddi_ficvfQ=noddi_ficvf[Q]
            dti_adQ=dti_ad[Q]
            dti_faQ=dti_fa[Q]
            dti_rdQ=dti_rd[Q]
            despot_pdQ=despot_pd[Q]
            despot_t1Q=despot_t1[Q]
            swanQ=swan[Q]

            obj3D[Q]=1
            obj3D[~Q]=0

            obj3D_dil=binary_dilation(obj3D).astype(obj3D.dtype)
            obj3D_ero=binary_erosion(obj3D).astype(obj3D.dtype)
            obj3D_bound_int=obj3D-obj3D_ero
            obj3D_bound_ext=obj3D_dil-obj3D

            try:
                t1QSbound_int=t1[obj3D_bound_int==1]
                t1QSbound_ext=t1[obj3D_bound_ext==1]
            except Exception as e:
                raise Exception('Failed for region %s, vol is %f'%(regs[regno],vol))
            
            f.write(output_format%(subjectID,regno,regs[regno],
                vol,
                vol/float(sumvol),
                np.nanmean(t1Q),
                np.nanstd(t1Q),
                np.nanmean(flairQ),
                np.nanstd(flairQ),
                np.nanmean(noddi_ficvfQ),
                np.nanstd(noddi_ficvfQ),
                np.nanmean(dti_adQ),
                np.nanstd(dti_adQ),
                np.nanmean(dti_faQ),
                np.nanstd(dti_faQ),
                np.nanmean(dti_rdQ),
                np.nanstd(dti_rdQ),
                np.nanmean(despot_pdQ),
                np.nanstd(despot_pdQ),
                np.nanmean(despot_t1Q),
                np.nanstd(despot_t1Q),
                np.nanmean(swanQ),
                np.nanstd(swanQ),
                np.nanmean(t1QSbound_int),
                np.nanstd(t1QSbound_int),
                np.nanmean(t1QSbound_ext),
                np.nanstd(t1QSbound_ext)
            ))

            if vol==0 or subject_type=='C':
                f.write(',0.0')
            elif subject_type=='N':
                f.write(',N')
            elif subject_type=='D':
                lmreg=lm[Q]
                f.write(',%f'%(np.sum(lmreg)))
            else:
                raise Exception('Unknown subject type '+subject_type)
            f.write('\n')

