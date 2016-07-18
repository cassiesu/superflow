#!/bin/bash

IMGDIR=data/image
FLOWDIR=data/flow
CODEDIR=code
IMG=$1

printf "\e[31mStart to make clean...\e[39m\n"
make -C ${CODEDIR} clean

printf "\n\e[31mStart to make...\e[39m\n"
make -C ${CODEDIR}

printf "\n\e[31mStart to run deepflow...\e[39m\n"
env CPUPROFILE="/Users/saiwenwang/Documents/College/ETH MSc 1/Fast Numerical Code/Project/tmp/deepflow.prof" ./${CODEDIR}/bin/deepflow ${IMGDIR}/${IMG}1.png ${IMGDIR}/${IMG}2.png ${FLOWDIR}/${IMG}_flow_current.flo

printf "\e[31mStart to diff...\e[39m\n"
xxd ${FLOWDIR}/${IMG}_flow_origin.flo > ${FLOWDIR}/${IMG}_flow_origin.hex
xxd ${FLOWDIR}/${IMG}_flow_current.flo > ${FLOWDIR}/${IMG}_flow_current.hex
diff ${FLOWDIR}/${IMG}_flow_origin.hex ${FLOWDIR}/${IMG}_flow_current.hex
