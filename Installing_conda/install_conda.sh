#!/bin/bash

cd /home/xzcaplas/PHAS0097_RLASSEN/salt/

source setup/setup_conda.sh

conda create --prefix /home//xzcaplas/PHAS0097_RLASSEN/salt/conda/envs/salt python=3.10
conda activate /home//xzcaplas/PHAS0097_RLASSEN/salt/conda/envs/salt

pip install jsonnet==0.17.0
pip install comet_ml
pip install -e .

conda deactivate

echo "Done."

