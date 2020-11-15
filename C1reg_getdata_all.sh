#!/bin/bash

# Epilepsy MRI-negative/positive lesion detection pipeline, Copyright 2020, University College London

for de in data_rigid_t1_space/*
do 
    if [ ! -f parcellation_data/parcellation_data_$(basename $de).csv ]; then
        ./C1reg_getdata.py $(basename $de)
    fi
done
