#!/bin/bash
# Config
SQUAD_DIR="squad"
# json_files
declare -a json_files=(train-v1.1.json  dev-v1.1.json)
# -----------------------------------------------------
download_squad_json() {
    json_fname=$1
    echo "Going to download ${json_fname} ..."
    curl -O https://rajpurkar.github.io/SQuAD-explorer/dataset/${json_fname}
}
# -----------------------------------------------------

# make the directory
mkdir -p "${SQUAD_DIR}"

# go to the squad dir
cd "${SQUAD_DIR}"
    # download them one by one
    for json in "${json_files[@]}"; do
        [[ -f $json ]]|| download_squad_json $json
    done
cd -
