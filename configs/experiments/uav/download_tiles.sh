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


if [ $# -lt 5 ]; then
    echo "Not enough arguments provided: bash download_tiles.sh <source> <dataset_name> <lat> <long> <radius> "
    exit 1
fi

SOURCE=$1
DATASET_NAME=$2
LAT=$3
LONG=$4
RADIUS=$5

# Go to map tools folder and download tiles.
cd ../../../map_tools
bash build.sh
bash run_container.sh downloader --lat $LAT --long $LONG --radius $RADIUS --map --size 1000 -o "temp_io/$DATASET_NAME/$SOURCE" --source $SOURCE
