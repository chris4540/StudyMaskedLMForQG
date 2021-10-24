#!/bin/bash
#       Environment for Windows
# Usage:
#   source <this script>

###### SCRIPT CONFIG ######
conda_script_dir="""
${HOME}/Anaconda3/Scripts
${HOME}/.anaconda3/bin
/opt/anaconda3/bin
"""
env_name="qgen"
###### SCRIPT CONFIG ######

if [[ ! -n "$(command -v conda)" ]]; then
    # loop over all possible directories
    for dir in ${conda_script_dir}; do
        activation_script=${dir}/activate
        if [[ -f ${activation_script} ]]; then
            echo "Found the env script:" "${activation_script}"
            source ${activation_script}
            break
        fi
    done

    # check if
    if [[ ! -n "$(command -v conda)" ]];then
        echo "ERROR: Please activate conda env and source this script!"
    fi
fi

echo "Going to activate the env: ${env_name}"
conda activate ${env_name}
