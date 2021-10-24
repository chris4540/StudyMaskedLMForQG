#!/bin/bash
#             Script documentation
# Available loc of tesla t4 GPU:
#   - europe-west2-a
#   - europe-west2-b
#   - europe-west3-b
#   - europe-west4-b
#   - europe-west4-c
#
# Available loc of tesla p100 GPU:
#   - europe-west1-b
#   - europe-west1-d
#   - europe-west4-a
# Available loc of tesla k80 GPU:
#   - europe-west1-b
#   - europe-west1-d
#
# Reference:
#   https://cloud.google.com/ai-platform/deep-learning-vm/docs/tensorflow_start_instance
# Location
#   https://cloud.google.com/compute/docs/gpus/
# Pricing
#   https://cloud.google.com/compute/gpus-pricing
# ----------------------------------------

# The name of the VM
export INSTANCE_NAME="tch14-t4"
export ZONE="europe-west4-b"
# Opt to make the VM PREEMPTIBLE or not; Either "true" or "false"
# Doc: https://cloud.google.com/compute/docs/instances/preemptible
export IS_PREEMPTIBLE_MACHINE=false
# ------------------------------------------------
# export IMAGE_NAME="pytorch-1-4-cu100-20200128"
export IMAGE_NAME="pytorch-1-4-cu101-v20201014-ubuntu-1804"
export INSTANCE_TYPE="n1-standard-2"
export GPU_CONFIG="type=nvidia-tesla-t4,count=1"
export BOOT_DISK_SIZE="200GB"
# export DISK_NAME=tch13-t4


args=(
  --machine-type=$INSTANCE_TYPE
  --zone=$ZONE
  --image=$IMAGE_NAME
  --image-project=deeplearning-platform-release
  --maintenance-policy=TERMINATE
  --boot-disk-size=$BOOT_DISK_SIZE
  --accelerator=$GPU_CONFIG
  --metadata="install-nvidia-driver=True,status-uptime-deadline=600.0"
)
if ${IS_PREEMPTIBLE_MACHINE}; then
  args+=( --preemptible )
fi

# print the create vm options out
echo "============================================="
echo VM instance name: "${INSTANCE_NAME}"
echo "--------------------------------------"
echo Create VM options:
for opt in "${args[@]}"; do
  echo ${opt}
done
echo "============================================="

echo "Creating VM...."
gcloud compute instances create ${INSTANCE_NAME} "${args[@]}"

