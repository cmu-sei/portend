#!/bin/bash
#
# Portend Toolset
# 
# Copyright 2024 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM24-1299
#


if [ $# -lt 3 ]; then
    echo "Not enough arguments provided: bash prep_dataset.sh <source> <dataset_name> <exp_folder>"
    exit 1
fi

SOURCE=$1
DATASET_NAME=$2
EXP_FOLDER=$3

# Call image prep.
bash prep_images.sh $SOURCE $DATASET_NAME $EXP_FOLDER

# Call map prep.
bash prep_map.sh $SOURCE $DATASET_NAME