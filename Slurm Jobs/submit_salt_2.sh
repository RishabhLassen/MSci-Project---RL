#!/bin/bash -l
# source /etc/slurm/gpu_variables.sh

#requesting one node

# keep environment variables


#SBATCH -N1 
#requesting cpus

# request enough memory
#SBATCH --mem=150G

#SBATCH --time=48:00:00

#SBATCH --export=ALL

# mail on failures
#SBATCH --mail-user=nikita.pond.18@ucl.ac.uk
#SBATCH --mail-type=FAIL

# Change log names; %j gives job id, %x gives job name
# SBATCH --output=/home/xzcaplas/PHAS0097_RLASSEN/salt/submit/output/slurm-%j.%x.out
# optional separate error output file
# SBATCH --error=/home/xzcaplas/PHAS0097_RLASSEN/salt/submit/output/slurm-%j.%x.err



# Request everything !
#SBATCH -c20
#SBATCH -p LIGHTGPU
#SBATCH --gres=gpu:a100
#SBATCH --job-name=base
log_out_file=/home/xzcaplas/PHAS0097_RLASSEN/submit/output/slurm-$SLURM_JOB_ID-$SLURM_JOB_NAME.out

# project=GN2.1_MC23_MC20
# project=testing
project=supervising/base
# project=HEPFORMER
name_base="gn2v01_phase2_without_hits"
config="/home/xzcaplas/PHAS0097_RLASSEN/salt/salt/configs/GN2.yaml"
output_dir="/home/xzcaplas/PHAS0097_RLASSEN/salt/submit/output/"$project
num_devices=1
num_workers=20
raw_batchsize=2000
batch_size=$(($raw_batchsize/$num_devices))
max_epochs=40

name=$name_base"_b"$raw_batchsize"_e"$max_epochs

node=$(hostname)
echo "Running on " $node

if [ "$node" == "compute-gpu-0-0.local" ]; then
    echo "Running on light GPU"
    source /home/xzcappon/.bashscripts/get_free_mig.bash

    # Check if a GPU was available
    if [ "$GPU_AVAILABLE" -eq 0 ]; then
        echo "No GPU is available, exiting."
        exit 1
    fi

    echo "CUDA_VISIBLE_DEVICES:" $CUDA_VISIBLE_DEVICES
    # export CUDA_VISIBLE_DEVICES=$dev
    # # speedup trick
    export OMP_NUM_THREADS=1

fi

# output_dir="/custom/dir"
# speedup trick
export OMP_NUM_THREADS=1

cd /home/xzcaplas/PHAS0097_RLASSEN/salt/
echo "Moved dir, now in:"
pwd

echo $CONDA_DEFAULT_ENV

echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES:"
echo $CUDA_VISIBLE_DEVICES

nvidia-smi

echo "Ntasks"
echo $SLURM_NTASKS

eval "$(/share/apps/anaconda/3-2022.05/bin/conda shell.bash hook)"
# activate environment
conda activate conda/envs/salt
source setup/install.sh

cd salt

export COMET_API_KEY=YMhHaYSPt1Ttem0UwiHAf81g2
export COMET_WORKSPACE='xzcaplas'
export COMET_PROJECT_NAME='salt-'$project
export OMP_NUM_THREADS=1

echo "Running training script..."
echo "Batchsize: "$batch_size
# python main.py -h
# TEST RUN 
# salt  fit --config $config --trainer.fast_dev_run 1 --trainer.devices 1 --data.num_workers 40 --data.batch_size 4000 --trainer.max_epochs 10 --trainer.default_root_dir $output_dir

# 3 GPU, 12k total batchsize, 10 epochs
# salt  fit --config $config --trainer.devices 3 --data.num_workers 120 --data.batch_size 4000 --trainer.max_epochs 10 --trainer.default_root_dir $output_dir

salt  fit --force --config $config --trainer.devices $num_devices --data.num_workers $num_workers \
    --data.batch_size $batch_size --trainer.default_root_dir $output_dir \
    --name $name --trainer.max_epochs $max_epochs \
    # --ckpt_path "/home/xzcappon/phd/tools/salt/salt/salt/logs/Overtraining/kfold/kfold_4fold_fold1_b12000_e40_20231009-T214438/ckpts/epoch=021-val_loss=0.57713.ckpt"
     
    # --trainer.fast_dev_run 1

# Ensure that we change config file, name of the submit script, 
# salt  fit --config $config --trainer.devices 2 --data.num_workers 80 --data.batch_size 2000 --trainer.max_epochs 40 --trainer.default_root_dir $output_dir

