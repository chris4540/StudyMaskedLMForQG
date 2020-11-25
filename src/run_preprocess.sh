#!/bin/bash
#
# Preprocessing script
#
# Step 1: patch the SQuAD file
# Step 2: make_triplet_txtds
# Step 3: cache preprcessed triplet text dataset to numpy compressed files
#  TODO: fix this script for step 1 and 2

#
# echo "Step 1: .............."
# python make_triplet_txtds.py
# mv prep_txt_ds/*.json ../txt_data/preprocessed/ -f
# echo "Step 2: ............."

echo "-------------------------------------------------------"
echo "Step 3: cache preprocesed text file to numpy data"
echo "-------------------------------------------------------"
python prep-cache_txtds.py
