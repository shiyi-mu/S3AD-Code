experiment_name="0414"
## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 
CUDA_VISIBLE_DEVICES=6
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0
cd ../

# CHECKPOINT_PATH=/server19/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/07-A3D-for-paper-r1/811-ar-nums-10-10/checkpoint/Stereo3D_latest.pth
CHECKPOINT_PATH=/server19/mushiyi/smb9_msy/02-Code/04-AD4AD/02-S3AD-public/S3AD-Code/workdirs/07-A3D-for-paper-r1/811-ar-nums-10-10/checkpoint/Stereo3D_latest.pth
echo $CHECKPOINT_PATH
./launchers/eval.sh 07-A3D-For_paper_r1/0-7-811-ar-train-arood-val-arnums-10-10.py $CUDA_VISIBLE_DEVICES $CHECKPOINT_PATH validation
