#!/usr/bin/env bash

# directory
output=~/Desktop/Research_3/pyDcop_playground/simulation
logdir=~/Desktop/Research_3/pyDcop_playground/log

mkdir -p $output
mkdir -p $logdir

# variables
log=${logdir}/metrics.csv

# generating data
colors_count=5
variables_count=10

if true; then
    for instance in {1..50}; do
        for delay in {0..10}; do
            echo running v$variables_count delay$delay insatnce${instance}
            pydcop solve --algo dpop \
                         --run_metric $logdir/gc_dpop_v${variables_count}_d${delay}_${instance}.csv \
                          --delay $delay \
                          ${input}/gc_v${variables_count}_${instance}.yaml \
                          > $logdir/gc_dpop_v${variables_count}_d${delay}_${instance}.yaml
#            pydcop solve --algo dsa \
#                         --algo_param stop_cycle:20 \
#                         --delay $delay \
#                         ${input}/graph_coloring_${variables_count}.yaml \
#                         > $logdir/gc_dsa_v${variables_count}_d${delay}.yaml
#            pydcop solve --algo mgm \
#                         --algo_params stop_cycle:20 \
#                         --delay $delay \
#                         ${input}/graph_coloring_${variables_count}.yaml \
#                         > $logdir/gc_mgm20_v${variables_count}_d${delay}.yaml
#            pydcop solve --algo maxsum \
#                         --delay $delay \
#                         ${input}/graph_coloring_${variables_count}.yaml \
#                         > $logdir/gc_maxsum_v${variables_count}_d${delay}.yaml
        done
    done
fi


# pydcop solve --algo mgm --algo_params stop_cycle:20 --collect_on cycle_change --run_metric $log  --delay $delay  $input &&

