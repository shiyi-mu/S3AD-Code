## run this if disparity map is needed, can be computed with point cloud or openCV BlockMatching
# You can run ./launcher/disparity_precompute.sh without arguments to see helper documents
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3

IsUsingPointCloud=0

cd ../

# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-201-ped-eval-stereo.py $IsUsingPointCloud
# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-202-ped-eval-mono.py $IsUsingPointCloud
# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-301-ar-train-arood-val-2d-only.py $IsUsingPointCloud
# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-302-ar-train-arood-val-2d3d.py $IsUsingPointCloud
# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-402-ped-eval-stereo-ped2d-gt.py $IsUsingPointCloud

# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-701-ar-train-arood-val-f_L.py $IsUsingPointCloud
# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-703-ar-train-arood-val-f_LS.py $IsUsingPointCloud

# ./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-811-ar-train-arood-val-arnums-10-10.py $IsUsingPointCloud

./launchers/disparity_precompute.sh 07-A3D-For_paper_r1/0-7-601-ar-train-arood-val-MSPF.py $IsUsingPointCloud
