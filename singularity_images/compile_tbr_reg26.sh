#!/bin/bash
export S26_PREFIX=/home/petr/util/s26
export SINGULARITY_TMPDIR=${S26_PREFIX}/tmp
export SINGULARITY_CACHEDIR=${S26_PREFIX}/cache
sudo nice -5 ${S26_PREFIX}/bin/singularity build /tmp/tbr_reg26.sif tbr_reg.def
