#!/bin/bash
# Analysis of a synthetic run against its truth (MF stage, sky.1): spectra per flux bin and component, beta vs radius,
# exact chi^2, flux split of the O1 regions, learned noise modes; summary in runs/synth_1deg/logs/an_<run>.log
# usage: analyze_run.sh RUN DATA_VARIANT TRUTH "COMPONENTS"   e.g. analyze_run.sh S1wfix_rpfl_t02 S1wfixlr_u50 1deg_main_sep_lr "o0 o1 bg"
S=$(cd "$(dirname "$0")" && pwd)
cd "$S/../.."
P=${PYTHON:-$(cd .. && pwd)/.pixi/envs/default/bin/python}
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=16
L=runs/synth_1deg/logs; R=runs/synth_1deg/r10_sf5_mf15; V=$1; D=$2; T=$3; O=$L/an_$V.log; : > $O
$P paper/synthetic/compare_synthetic.py --run $R/$V --sky sky.1 --truth $T --label $V > $L/compare_$V.log 2>&1; echo "compare exit $?" >> $O
grep -E "^flux" $L/compare_$V.log | grep -E "alpha_model|beta_model" >> $O
for c in $4; do $P paper/synthetic/compare_component.py --run $R/$V --truth $T --comp $c 2>&1 | grep -E "pixels with|I0 \[|alpha \(|beta \(|Error" | sed "s/^/$c: /" >> $O; done
$P paper/synthetic/check_beta_radius.py --run $R/$V --truth $T --variant $D > $L/check_beta_radius_$V.log 2>&1; echo "radius exit $?" >> $O
grep -A11 "^region" $L/check_beta_radius_$V.log >> $O
$P $S/exact_chi2_rec.py $R/$V $T 2>&1 | grep -vE "^(WARNING|INFO)" | tail -8 >> $O
$P $S/comp_split.py $PWD/$R/$V $T 2>&1 | grep -E "SPLIT|Error" >> $O
$P $S/nm_modes.py $PWD/$R/$V 2>&1 | tail -14 >> $O
echo finished >> $O
