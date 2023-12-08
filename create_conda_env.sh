WORKSPACE_DIR=$HOME/workspace/tmp/anonymous-alberdice  # change this accordingly

ENV_NAME=alberdice

# for gftooball
sudo apt update
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
#

conda create -n $ENV_NAME -c pytorch python=3.8 pytorch=1.11.0 cudatoolkit=11.3.1 cudnn=8.2 pip
conda activate $ENV_NAME

PIP_BIN=$HOME/miniconda3/envs/$ENV_NAME/bin/pip3

$PIP_BIN install sacred wandb matplotlib tqdm jinja2 traitlets gym==0.21.0 rware==1.0.3

# for VecEnv
sudo apt install libx11-dev
conda install cython
$PIP_BIN install stable-baselines3[extra]

cd $WORKSPACE_DIR/marl_env
$PIP_BIN install -e .
cd $WORKSPACE_DIR

conda install -c conda-forge gcc=12.1.0 py-boost

# Refer to https://github.com/google-research/football for required packages
# pip install gfootball
