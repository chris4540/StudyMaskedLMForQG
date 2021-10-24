#!/bin/bash
#
#                       Script docuementation
#  This script is to build an environment for developing or deployment.
#  For installing package on gcp instances, use pip install -r requirements.txt
#
set -e


# We use "$@" instead of $* to preserve argument-boundary information
opts=$(getopt -o 'hd' --long 'help,dev' -- "$@") || exit
## Print help
function show_help {
    echo "Usage:  $BASH_SOURCE [-d|--dev] --name <env_name>"
    echo ""
    echo "          -d, --dev create the environment in dev mode"
    echo "          --name  create the environment with the name specified"

}

# Preset option val
make_dev_env=false
for opts; do
    echo $1
    case "$1" in
    (-h|--help) show_help; exit 0 ;;
    (-d|--dev) make_dev_env=true;;
    (--name)
        # handle optional: getopt normalizes it into an empty string
        if [ -n "$2" ]; then
            env_name=$2; shift;
        fi;;
    (--)  break;;                # end
    (*)   show_help; exit 1;;           # error
    esac
    shift
done

if $make_dev_env; then
    echo "Creating the env in dev mode..."
fi
if [[ -z ${env_name} ]]; then
    env_name="qgen"
fi
echo "The environment name: ${env_name}"

#--------------------------------------------------------------------
###### SCRIPT CONFIG ######
possible_conda_dir="""
${HOME}/Anaconda3/Scripts
${HOME}/.anaconda3/bin
"""
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
###### SCRIPT CONFIG ######
for dir in ${possible_conda_dir}; do
    activation_script=${dir}/activate
    if [[ -f ${activation_script} ]]; then
        echo "Going to source the following conda script: ${activation_script}"
        source ${activation_script}
        break
    fi
done
#--------------------------------------------------------------------

# delete the env with:
conda env remove -n ${env_name}

# make the env
conda create -y --name ${env_name} python=3.7

# activate it
conda activate ${env_name}

# Install pytorch
conda install -y pytorch torchvision cpuonly -c pytorch
# conda install  pytorch torchvision cudatoolkit=10.1 -c pytorch

# stanza; for spliting sentences
pip install stanza

pip install -r ${script_dir}/../requirements.txt


if $make_dev_env; then
    pip install autopep8

    pip install pylint

    pip install mypy

    pip install flake8
fi
