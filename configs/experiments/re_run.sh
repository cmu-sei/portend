#!/usr/bin/env bash
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


# Stop on error.
set -e

# Check number of arguments.
if [ $# -ne 2 ]; then
    echo "Runtime or experiments result path is missing."
    echo "Proper way to call: bash re_run.sh <script_to_run> <exp_results_path>"
    exit 1
fi

# Check and get arguments.
SCRIPT=$1
if [ "$SCRIPT" != "run_local.sh" ] && [ "$SCRIPT" != "run_container.sh" ]; then
    echo "Script to run can only be run_local.sh, or run_container.sh"
    exit 1
fi
EXP_RESULTS_PATH="$2"

# Move to base folder.
cd ../../

# Get the current date and time to use as prefix for results.
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
date_time_string="$current_date-$current_time"

# Run predictor on existing results.
echo "Executing Predictor on results from: $EXP_RESULTS_PATH"
for exp_folder in ./"$EXP_RESULTS_PATH"/*; do
    [ -e "$exp_folder" ] || continue

    if [ -d "$exp_folder" ]; then
        echo "Found folder $exp_folder"
    else
        continue
    fi

    prefix="$date_time_string/rerun-$(basename $exp_folder)"

    # Get the predictor config.
    pred_folder="$exp_folder/predictor_config/"
    for config in "$pred_folder"/*; do
        echo $config
        pred_config=$config
    done

    # Get the drift files, just to be copied over.
    drifts_folder="$exp_folder/drift_config/"
    drift_files=""
    for config in "$drifts_folder"/*; do
        echo $config
        drift_files+=" $config"
    done    

    echo "  "
    echo "---------------------------------------"
    echo $prefix
    echo "Executing Predictor using: $pred_config"
    bash $SCRIPT predictor --pack "$prefix" -c "$pred_config" --pfolder "$exp_folder" --drifts $drift_files
done
