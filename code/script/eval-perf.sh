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

	printf "\n\e[31mCPU cycle: $IMAGE_SIZE...\e[39m\n"
	perf record -e cpu-cycles ./${CODEDIR}/bin/deepflow ${IMGDIR}/${IMG}1_${IMAGE_SIZE}.png ${IMGDIR}/${IMG}2_${IMAGE_SIZE}.png ${FLOWDIR}/${IMG}_${IMAGE_SIZE}_flow_current.flo
	echo -e "\a"
	perf report

	printf "\n\e[31mL1-load-miss: $IMAGE_SIZE...\e[39m\n"
	perf record -e L1-dcache-load-misses ./${CODEDIR}/bin/deepflow ${IMGDIR}/${IMG}1_${IMAGE_SIZE}.png ${IMGDIR}/${IMG}2_${IMAGE_SIZE}.png ${FLOWDIR}/${IMG}_${IMAGE_SIZE}_flow_current.flo
	echo -e "\a"
	perf report

	printf "\n\e[31mL1-store-miss: $IMAGE_SIZE...\e[39m\n"
	perf record -e L1-dcache-store-misses ./${CODEDIR}/bin/deepflow ${IMGDIR}/${IMG}1_${IMAGE_SIZE}.png ${IMGDIR}/${IMG}2_${IMAGE_SIZE}.png ${FLOWDIR}/${IMG}_${IMAGE_SIZE}_flow_current.flo
	echo -e "\a"
	perf report

	let "IMAGE_SIZE*=2"
done