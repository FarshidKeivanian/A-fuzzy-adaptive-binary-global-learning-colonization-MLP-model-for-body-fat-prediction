#!/bin/sh
#
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -l walltime=4:00:00
#PBS -k oe

cd $PBS_O_WORKDIR

source /etc/profile.d/modules.sh
module load matlab/R2017b

matlab -singleCompThread -nodisplay -r "FABEICA;exit;"


