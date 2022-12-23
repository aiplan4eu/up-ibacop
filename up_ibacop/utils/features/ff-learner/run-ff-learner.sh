#! /bin/sh
echo "Tiempo limite" 

ulimit -t $5
ulimit -v 3145728

# domaio problema algoritmo(14) Learner

NBOUND=100000

if [ $# -lt 6 ]; then
    ALG=0
else
    ALG=$6
fi
    
echo "Ejecutando alg: "
echo $ALG

case $ALG in 
    0) system/ff-learner/ff-learner -p $1 -o $2 -f $3 -k $1roller/ -S 14 -L $4 
	;;
    1) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -S 14 -L $4
	;;
    2) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -S 14 -R -L $4
	;;
     # See algorithm alternatives    
    # With repeated lists of nodes
    3) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -Y -A -N $NBOUND
	;;
    4) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -Y -A
	;;
    5) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -Y -N $NBOUND
	;;
    6) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -Y
	;;
    # without keeping repeated lists
    7) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -A -N $NBOUND
	;;
    8) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -A
	;;
    9) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -N $NBOUND
	;;
    10) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 
	;;
    # without repeated consideration
    11) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -R -A -N $NBOUND
	;;
    12) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -R -A
	;;
    13) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -R -N $NBOUND
	;;
    14) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -R
	;;
    # WBFS with repeated list of nodes
    15) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3 -Y -N $NBOUND
	;;
    16) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3 -Y
	;;
    17) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3 -N $NBOUND
	;;
    18) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3
	;;
    19) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3 -R -N $NBOUND
	;;
    20) system/ff-learner/roller3.0 -p $1 -o $2 -f $3 -k $7 -L $4 -S 14 -h 3 -R
	;;

    
esac

