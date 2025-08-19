#!/bin/bash
set -e
# CUDA_PATH=/usr/local/cuda #cuda: /usr/local/cuda -> /usr/local/cuda
# CUDA_VER=$(cat $CUDA_PATH/version.txt | cut -d ' ' -f 3)
# export LD_LIBRARY_PATH="/usr/local/cuda-11.1/"  
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# MAX_GCC_VERSION=9
TORCH_VER=$(python3 -c "import torch;print(torch.__version__)")
CUDA_VER=$(python3 -c "import torch;print(torch.version.cuda)")

if [[ $CUDA_VER < "10.0" ]] ; then 
    echo "The current version of pytorch/cuda is $TORCH_VER/$CUDA_VER which could be not compatible with deformable convolution, we will not compile DCN for now. As long as you do not init DCN instance in code, the code will run fine."
else
    pushd visualDet3D/networks/lib/ops/dcn
    sh make.sh
    rm -r build
    popd

    pushd visualDet3D/networks/lib/ops/iou3d
    sh make.sh
    rm -r build
    popd
fi
