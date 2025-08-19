experiment_name="0414"
## train the model with one GPU
# You can run ./launcher/train.sh without arguments to see helper documents
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
 
CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0
cd ../
# 4 of 10
# CHECKPOINT_PATH=/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/05-A3D-for-paper/0-4-4-Kitti-animal-eval-ar_nums4-10/checkpoint/Stereo3D_latest.pth

# 6 of 10
# CHECKPOINT_PATH=/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/05-A3D-for-paper/0-4-6-Kitti-animal-eval-ar_nums6-10/checkpoint/Stereo3D_latest.pth
# 8 of 10
# CHECKPOINT_PATH=/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/05-A3D-for-paper/0-4-8-Kitti-animal-eval-ar_nums8-10/checkpoint/Stereo3D_latest.pth
# all ar
# CHECKPOINT_PATH=/data3/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/05-A3D-for-paper/02-kitti-animal_eval/checkpoint/Stereo3D_latest.pth

CHECKPOINT_PATH=/server19/mushiyi/smb9_msy/02-Code/04-AD4AD/02-Ood3D/workdirs/07-A3D-for-paper-r1/01-kitti-base/checkpoint/Stereo3D_latest.pth
echo $CHECKPOINT_PATH
./launchers/eval.sh 07-A3D-For_paper_r1/0-7-001-kitti-animal_final.py $CUDA_VISIBLE_DEVICES $CHECKPOINT_PATH validation
