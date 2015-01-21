#!/bin/bash
#executable names MUST be compiled so they end in .ARCH (uname -p), for example: "a.out.x86_x64" or "a.x86_x64" or "nbody.x86_x64"
#in case your architecture/OS doesn't support the "uname -p" command, you can setup a name for this architecture by giving a value manually
CURR_ARCH=`uname -p`
#if -p didnt return a value
if [ "$CURR_ARCH" = "unknown" ]; then CURR_ARCH=`uname -m`; fi
if [ "$NX_OFFL_DEBUG" = "1" ]; 
then 
    echo "NX_OFFL_DEBUG: Allocating offload worker at host "`hostname `
elif [ "$NX_OFFL_DEBUG" = "2" ]; 
then 
    echo "NX_OFFL_DEBUG: Allocating offload worker at host \""`hostname `"\", architecture" $CURR_ARCH 
fi
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

#undef OMP num threads and NX SMP WORKERS, if they are defined in the host, most likely we don't want them defined in the remote device
unset OMP_NUM_THREADS
unset NX_SMP_WORKERS
#export variables which are process/host dependant
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

if [ "$NX_OFFL_DEBUG" = "4" ]; 
then 
    echo "NX_OFFL_DEBUG: Allocating offload worker at host \""`hostname `"\" with "`taskset -cp $$`", architecture" $CURR_ARCH 
fi
#PROCESS OFFL_* OR $ARCH_* VARIABLES TO *
OFFLOAD_VARS=${!OFFL_@}
for OFFLOAD_CURR_VAR in $OFFLOAD_VARS
do
	eval VALUE_OFFLOAD_CURR_VAR=\$$OFFLOAD_CURR_VAR
	NEW_OFFLOAD_CURR_VAR=${OFFLOAD_CURR_VAR#OFFL_}
	export $NEW_OFFLOAD_CURR_VAR="$VALUE_OFFLOAD_CURR_VAR"
   if [ "$NX_OFFL_DEBUG" = "4" ]; 
   then 
     echo Converting environment variable $OFFLOAD_CURR_VAR to $NEW_OFFLOAD_CURR_VAR, value $VALUE_OFFLOAD_CURR_VAR
   fi
done
AUXX_TMP_VARS='OFFLOAD_VARS=${!'$LOWER_CURR_ARCH'_@}'
eval $AUXX_TMP_VARS
for OFFLOAD_CURR_VAR in $OFFLOAD_VARS
do
	eval VALUE_OFFLOAD_CURR_VAR=\$$OFFLOAD_CURR_VAR
	NEW_OFFLOAD_CURR_VAR=${OFFLOAD_CURR_VAR#${LOWER_CURR_ARCH}_}
	export $NEW_OFFLOAD_CURR_VAR="$VALUE_OFFLOAD_CURR_VAR"
   if [ "$NX_OFFL_DEBUG" = "4" ]; 
   then 
     echo Converting environment variable $OFFLOAD_CURR_VAR to $NEW_OFFLOAD_CURR_VAR, value $VALUE_OFFLOAD_CURR_VAR
   fi
done
AUXX_TMP_VARS='OFFLOAD_VARS=${!'$UPPER_CURR_ARCH'_@}'
eval $AUXX_TMP_VARS
for OFFLOAD_CURR_VAR in $OFFLOAD_VARS
do
	eval VALUE_OFFLOAD_CURR_VAR=\$$OFFLOAD_CURR_VAR
	NEW_OFFLOAD_CURR_VAR=${OFFLOAD_CURR_VAR#${UPPER_CURR_ARCH}_}
	export $NEW_OFFLOAD_CURR_VAR="$VALUE_OFFLOAD_CURR_VAR"
   if [ "$NX_OFFL_DEBUG" = "4" ]; 
   then 
     echo Converting environment variable $OFFLOAD_CURR_VAR to $NEW_OFFLOAD_CURR_VAR, value $VALUE_OFFLOAD_CURR_VAR
   fi
done
#END PROCESS OFFL_* OR ARCH_* VARIABLES TO *

export OMPSS_OFFLOAD_SLAVE=1
if [ "$NX_OFFL_DEBUG" = "3" ]; 
then 
    echo "NX_OFFL_DEBUG: Allocating offload worker at host \""`hostname `"\" with "`taskset -cp $$`", architecture" $CURR_ARCH 
fi
if [ "$NX_OFFL_DEBUG" = "5" ]; 
then 
   echo "NX_OFFL_DEBUG: Allocating offload worker at host \""`hostname `"\" with "`taskset -cp $$`
	echo "Launching exec $filename.$CURR_ARCH $ARG1 $ARG2"
	env
fi
if [ ! -f $filename.$CURR_ARCH ]; then
echo "WARNING: By convention, when offloading, your executable should be named as \"$filename.$CURR_ARCH\" if you want to offload to this architecture (NAME.ARCHITECTURE). Falling back to master executable ($original_filename)"
exec $original_filename $ARG1 $ARG2
else
exec $filename.$CURR_ARCH $ARG1 $ARG2
fi