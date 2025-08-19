experiment_name="0414"
## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 
CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0
cd ../

CHECKPOINT_PATH=/server19/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/05-A3D-for-paper/0-2-2-Kitti-animal-eval_kitti_23d_indoor2d/checkpoint/Stereo3D_29.pth
echo $CHECKPOINT_PATH
./launchers/eval.sh  07-A3D-For_paper_r1/0-7-301-ar-train-arood-val-2d-only.py  $CUDA_VISIBLE_DEVICES $CHECKPOINT_PATH validation
