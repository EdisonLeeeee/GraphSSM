#!/bin/bash

ssm=siso
device=0

# DBLP-3
python graphssm/ssm_main.py --hidden_channels 64 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset dblp3 \
                            --weight_decay 0.0001 \
                            --device $device

# Brain
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset brain \
                            --weight_decay 0.0001 \
                            --d_state 32 \
                            --device $device
# Reddit
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset reddit \
                            --device $device

# DBLP=10
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer interp \
                            --dataset dblp10 \
                            --device $device

# Tmall
python graphssm/ssm_tmall.py --hidden_channels 32 \
                             --ssm_format $ssm \
                             --token_mixer interp \
                             --device $device

# arXiv
python graphssm/ssm_arxiv.py --hidden_channels 64 \
                             --ssm_format $ssm \
                             --token_mixer interp \
                             --device $device