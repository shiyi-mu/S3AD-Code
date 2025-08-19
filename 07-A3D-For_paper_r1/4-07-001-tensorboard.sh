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
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ../
log_path='workdirs/07-A3D-for-paper-r1/201-ped-eval-stereo/log/0414config=07-A3D-For_paper_r1/0-7-201-ped-eval-stereo.py/'
tensorboard --logdir=$log_path --bind_all