#!/bin/bash
#executable names MUST be compiled so they end in .ARCH (uname -p), for example: "a.out.x86_x64" or "a.x86_x64" or "nbody.x86_x64"
#in case your architecture/OS doesn't support the "uname -p" command, you can setup a name for this architecture by giving a value manually
CURR_ARCH=`uname -p`
#if -p didnt return a value
if [ "$CURR_ARCH" = "unknown" ]; then CURR_ARCH=`uname -m`; fi
#remove everything after last point (get the original filename without arch)
original_filename=$1
filename=${1%.*}
#in case you want to "rename"/add additional names for an architecture, follow this schema
if [ "$CURR_ARCH" = "k1om" ] && [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH="mic"; fi
if [ "$CURR_ARCH" = "x86_64" ] && [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH="intel64"; fi
LOWER_CURR_ARCH=`echo $CURR_ARCH | tr '[:upper:]' '[:lower:]'`
UPPER_CURR_ARCH=`echo $CURR_ARCH | tr '[:lower:]' '[:upper:]'`
if [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH=$UPPER_CURR_ARCH; fi
ARG1=$1
ARG2=$2
shift
shift

ARCH_NUM_THREADS=$UPPER_CURR_ARCH"_OMP_NUM_THREADS"
LOWER_ARCH_NUM_THREADS=$LOWER_CURR_ARCH"_OMP_NUM_THREADS"

ARCH_NX_SMP_WORKERS=$UPPER_CURR_ARCH"_NX_SMP_WORKERS"
LOWER_ARCH_NX_SMP_WORKERS=$LOWER_CURR_ARCH"_NX_SMP_WORKERS"


#################### OMP BLOCK
unset OMP_NUM_THREADS

#OMP_NUM_THREADS ignored, OFFLOAD_OMP_NUM_THREADS used
if [ "x$OFFL_OMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$OFFL_OMP_NUM_THREADS
fi
unset OFFL_OMP_NUM_THREADS

eval TMP_NUM_THREADS=\$$ARCH_NUM_THREADS
#OMP_NUM_THREADS ignored, $ARCH_OMP_NUM_THREADS used
if [ "x$TMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$TMP_NUM_THREADS
fi
unset $ARCH_NUM_THREADS
unset TMP_NUM_THREADS
eval TMP_NUM_THREADS=\$$LOWER_ARCH_NUM_THREADS
#OMP_NUM_THREADS ignored, $ARCH_OMP_NUM_THREADS used
if [ "x$TMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$TMP_NUM_THREADS
fi
unset $LOWER_ARCH_NUM_THREADS
unset TMP_NUM_THREADS


#################### END OMP BLOCK

#################### NX_SMP BLOCK
unset NX_SMP_WORKERS

#OMP_NUM_THREADS ignored, OFFLOAD_OMP_NUM_THREADS used
if [ "x$OFFL_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$OFFL_NX_SMP_WORKERS
fi
unset OFFL_NX_SMP_WORKERS

eval TMP_NX_SMP_WORKERS=\$$ARCH_NX_SMP_WORKERS
#NX_SMP_WORKERS ignored, $ARCH_NX_SMP_WORKERS used
if [ "x$TMP_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$TMP_NX_SMP_WORKERS
fi
unset $ARCH_NX_SMP_WORKERS
unset TMP_NX_SMP_WORKERS
eval TMP_NX_SMP_WORKERS=\$$LOWER_ARCH_NX_SMP_WORKERS
#NX_SMP_WORKERS ignored, $ARCH_NX_SMP_WORKERS used
if [ "x$TMP_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$TMP_NX_SMP_WORKERS
fi
unset $LOWER_ARCH_NX_SMP_WORKERS
unset TMP_NX_SMP_WORKERS


#################### END NX_SMP

second="="
first=${@}
first=${first//EQUAL/$second}
export "${first}" 
if [ "x$TASKSET" != "x" ]; then 
	taskset -cp $TASKSET $$ > /dev/null 2>&1
elif [ "x$NX_BINDING_START" == "x" ]; then
   #thread 0 on MIC is last core so we start on first core by default
   export NX_BINDING_START=1
fi

#################### OMP BLOCK

#OMP_NUM_THREADS ignored, OFFLOAD_OMP_NUM_THREADS used
if [ "x$OFFL_OMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$OFFL_OMP_NUM_THREADS
fi
unset OFFL_OMP_NUM_THREADS

eval TMP_NUM_THREADS=\$$ARCH_NUM_THREADS
#OMP_NUM_THREADS ignored, $ARCH_OMP_NUM_THREADS used
if [ "x$TMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$TMP_NUM_THREADS
fi
unset $ARCH_NUM_THREADS
unset TMP_NUM_THREADS
eval TMP_NUM_THREADS=\$$LOWER_ARCH_NUM_THREADS
#OMP_NUM_THREADS ignored, $ARCH_OMP_NUM_THREADS used
if [ "x$TMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$TMP_NUM_THREADS
fi
unset $LOWER_ARCH_NUM_THREADS
unset TMP_NUM_THREADS

#################### END OMP BLOCK

#################### NX_SMP BLOCK

#OMP_NUM_THREADS ignored, OFFLOAD_OMP_NUM_THREADS used
if [ "x$OFFL_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$OFFL_NX_SMP_WORKERS
fi
unset OFFL_NX_SMP_WORKERS

eval TMP_NX_SMP_WORKERS=\$$ARCH_NX_SMP_WORKERS
#NX_SMP_WORKERS ignored, $ARCH_NX_SMP_WORKERS used
if [ "x$TMP_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$TMP_NX_SMP_WORKERS
fi
unset $ARCH_NX_SMP_WORKERS
unset TMP_NX_SMP_WORKERS
eval TMP_NX_SMP_WORKERS=\$$LOWER_ARCH_NX_SMP_WORKERS
#NX_SMP_WORKERS ignored, $ARCH_NX_SMP_WORKERS used
if [ "x$TMP_NX_SMP_WORKERS" != "x" ]; then 
	export NX_SMP_WORKERS=$TMP_NX_SMP_WORKERS
fi
unset $LOWER_ARCH_NX_SMP_WORKERS
unset TMP_NX_SMP_WORKERS

#################### END NX_SMP

export OMPSS_OFFLOAD_SLAVE=1
if [ ! -f $filename.$CURR_ARCH ]; then
echo "WARNING: By convention, when offloading, your executable should be named as \"$filename.$CURR_ARCH\" if you want to offload to this architecture (NAME.ARCHITECTURE). Falling back to master executable ($original_filename)"
exec $original_filename $ARG1 $ARG2
else
exec $filename.$CURR_ARCH $ARG1 $ARG2
fi
