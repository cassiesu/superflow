#!/bin/bash

IMGDIR=data/ship-set
FLOWDIR=data/flow
CODEDIR=code
IMG=ship

printf "\e[31mStart to make clean...\e[39m\n"
make -C ${CODEDIR} clean

printf "\n\e[31mStart to make...\e[39m\n"
make -C ${CODEDIR}

IMAGE_SIZE=64

for i in `seq 6 12`;
do
	printf "\n\e[31mStart to run deepflow: $IMAGE_SIZE...\e[39m\n"
	env CPUPROFILE="/Users/saiwenwang/Documents/College/ETH MSc 1/Fast Numerical Code/Project/tmp/deepflow.prof" ./${CODEDIR}/bin/deepflow ${IMGDIR}/${IMG}1_${IMAGE_SIZE}.png ${IMGDIR}/${IMG}2_${IMAGE_SIZE}.png ${FLOWDIR}/${IMG}_${IMAGE_SIZE}_flow_current.flo
	let "IMAGE_SIZE*=2"
done
