#!/usr/bin/env bash

export CONDA_PREFIX=/mnt/niumiaohe/miniconda3/envs/elf
export CONDA_DEFAULT_ENV=elf
export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH

# make prompt visibly show env name
case "$PS1" in
  "(elf) "*) ;;
  *) export PS1="(elf) $PS1" ;;
esac

echo "Activated visible env: elf"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "python=$(which python)"
echo "pip=$(which pip)"
echo "torchrun=$(which torchrun)"
