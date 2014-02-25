#/bin/bash
#executable names MUST be compiled so they end in .ARCH (uname -p), for example: "a.out.x86_x64" or "a.x86_x64" or "nbody.x86_x64"
#in case your architecture/OS doesn't support the "uname -p" command, you can setup a name for this architecture by giving a value manually
CURR_ARCH=`uname -p`
#if -p didnt return a value
if [ "$CURR_ARCH" = "unknown" ]; then CURR_ARCH=`uname -m`; fi
#remove everything after last point (get the original filename without arch)
filename=${1%.*}
#in case you want to "rename"/add additional names for an architecture, follow this schema
if [ "$CURR_ARCH" = "k1om" ] && [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH="mic"; fi
if [ "$CURR_ARCH" = "x86_64" ] && [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH="intel64"; fi
ARG1=$1
ARG2=$2
shift
shift
#OMP_NUM_THREADS ignored, MIC_OMP_NUM_THREADS used
unset OMP_NUM_THREADS 
if [ "x$MIC_OMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$MIC_OMP_NUM_THREADS
   unset NX_THREADS
fi
unset MIC_OMP_NUM_THREADS
export ${@} 
if [ "x$TASKSET" != "x" ]; then 
	taskset -cp $TASKSET $$ > /dev/null 2>&1
elif [ "x$NX_BINDING_START" == "x" ]; then
   #thread 0 on MIC is last core so we start on first core by default
   export NX_BINDING_START=1
fi
#After we exported user-defined environment vars, use MIC_OMP_NUM_THREADS again
if [ "x$MIC_OMP_NUM_THREADS" != "x" ]; then 
	export OMP_NUM_THREADS=$MIC_OMP_NUM_THREADS
   unset NX_THREADS
fi
export NX_ARGS=$NX_ARGS" --spins=1 --sleep-time=10 --sleeps=50000000"
export OMPSS_OFFLOAD_SLAVE=1
exec $filename.$CURR_ARCH $ARG1 $ARG2