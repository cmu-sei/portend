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
if [ $# -ne 3 ]; then
    echo "Runtime, drift config, or predictor config missing."
    echo "Proper way to call: bash exp_run.sh <script_to_run> <exp_gen_config_path> <predictor_config_path>"
    echo "Where <script_to_run> can be run_local.sh to run locally, or run_container.sh to run in a container."
    exit 1
fi

# Base folder for experiment configs.
EXP_BASE_FOLDER="configs/experiments"

# Base folder for generated drift configs.
GENERATED_CONFIGS_FOLDER="$EXP_BASE_FOLDER/temp_configs"

# Check and get arguments.
SCRIPT=$1
if [ "$SCRIPT" != "run_local.sh" ] && [ "$SCRIPT" != "run_container.sh" ]; then
    echo "Script to run can only be run_local.sh, or run_container.sh"
    exit 1
fi
DRIFT_GEN_CONFIG="$EXP_BASE_FOLDER/$2"
PRED_CONFIG="$EXP_BASE_FOLDER/$3"

# Move to base folder.
cd ../../

# Generate all configs.
echo "Executing experiment drift config generator: $DRIFT_GEN_CONFIG"
bash $SCRIPT helper -c $DRIFT_GEN_CONFIG -o $GENERATED_CONFIGS_FOLDER

# Get the current date and time to use as prefix for results.
current_date=$(date +"%Y-%m-%d")
current_time=$(date +"%H-%M-%S")
date_time_string="$current_date-$current_time"

# Run drifter and predictor for each config.
echo "Executing Drifter and Predictor on files on: $GENERATED_CONFIGS_FOLDER"
for drift_config_filename in ./"$GENERATED_CONFIGS_FOLDER"/*.json; do
    [ -e "$drift_config_filename" ] || continue

    echo "Executing Drifter using: $drift_config_filename"
    bash $SCRIPT drifter -c "$drift_config_filename" && \

    name_only="${drift_config_filename##*/}"
    prefix="${name_only%.*}"
    prefix="$date_time_string/exp-$prefix"

    echo "  "
    echo "---------------------------------------"
    echo $prefix
    echo "Executing Predictor using: $PRED_CONFIG for drift from $drift_config_filename"
    bash $SCRIPT predictor --pack "$prefix" -c "$PRED_CONFIG" --drifts "$DRIFT_GEN_CONFIG" "$drift_config_filename"
done
