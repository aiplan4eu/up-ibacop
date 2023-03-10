#!/bin/bash
BASEDIR="$(dirname "$0")"
python2.7 "$BASEDIR/translate/translate.py" $1  $2 
"$BASEDIR/preprocess/preprocess" < heuristics.sas
(ulimit -t 100;"$BASEDIR/search/downward" --search "eager_greedy([add,blind,cg,cea,ff,goalcount,lmcount(lm_rhw(reasonable_orders=true,lm_cost_type=2,cost_type=2)),lmcut,hmax])" < heuristic;)
