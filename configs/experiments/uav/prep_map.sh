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


if [ $# -lt 2 ]; then
    echo "Not enough arguments provided: bash prep_map.sh <source> <dataset_name>"
    exit 1
fi

SOURCE=$1
DATASET_NAME=$2

# Copy map to map folder.
echo "Preparing map folder at process_io/$DATASET_NAME/map"
cd ../../../
rm -r "process_io/$DATASET_NAME/map"
mkdir "process_io/$DATASET_NAME"
mkdir "process_io/$DATASET_NAME/map"

echo "Copying map to map folder."
cp ./map_tools/temp_io/$DATASET_NAME/$SOURCE/map/* process_io/$DATASET_NAME/map
rm "process_io/$DATASET_NAME/map/"*combined.*
echo "Finished copying map and deleting combined version."
