## Compute image database and anchors mean/std
# You can run ./launcher/det_precompute.sh without arguments to see helper documents
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=1

cd  ../
# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-201-ped-eval-stereo.py train
# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-202-ped-eval-mono.py train

# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-301-ar-train-arood-val-2d-only.py train
# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-302-ar-train-arood-val-2d3d.py train

# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-402-ped-eval-stereo-ped2d-gt.py train

# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-701-ar-train-arood-val-f_L.py train
# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-703-ar-train-arood-val-f_LS.py train

# bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-811-ar-train-arood-val-arnums-10-10.py  train

bash launchers/det_precompute.sh 07-A3D-For_paper_r1/0-7-811-ar-train-arood-val-arnums-10-10.py train
