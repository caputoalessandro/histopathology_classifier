GPU='1'
MICRON=800
NAME=$MICRON"_baseline_lh"

$HOME/histopatho-cancer-grading/scripts/run.sh models/run_experiment.py $NAME $GPU  \
-co=1 -lr=1e-1 -wd=1e-4 -mm=0.9 -lrs=1 -e=60 -rn=$NAME -wdb=1  \
-tf=0 -nc=2 -m=$MICRON -fd=800 --tsv="" -w=0 -o="sgd"

