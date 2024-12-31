#!/bin/bash

ssm=siso
device=0

# DBLP-3
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset dblp3 \
                            --device $device \
                            --weight_decay 0.01 \
                            --model_name s6
                            

# Brain
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset brain \
                            --device $device \
                            --weight_decay 0.01 \
                            --model_name s6
# Reddit
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer conv1d \
                            --dataset reddit \
                            --device $device \
                            --model_name s6  

# DBLP=10
python graphssm/ssm_main.py --hidden_channels 32 \
                            --ssm_format $ssm \
                            --token_mixer interp \
                            --dataset dblp10 \
                            --device $device \
                            --model_name s6  

# Tmall
python graphssm/ssm_tmall.py --hidden_channels 32 \
                             --ssm_format $ssm \
                             --token_mixer interp \
                             --device $device \
                             --model_name s6  

# arXiv
python graphssm/ssm_arxiv.py --hidden_channels 64 \
                             --ssm_format $ssm \
                             --token_mixer interp \
                             --device $device \
                             --model_name s6  