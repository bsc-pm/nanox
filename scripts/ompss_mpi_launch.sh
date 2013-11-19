#/bin/bash
#executable names MUST be compiled so they end in .ARCH (uname -p), for example: "a.out.x86_x64" or "a.x86_x64" or "nbody.x86_x64"
#in case your architecture/OS doesn't support the "uname -p" command, you can setup a name for this architecture by giving a value manually
CURR_ARCH=`uname -p`
#remove everything after last point (get the original filename without arch)
filename=${1%.*}
#in case you want to "rename"/add additional names for an architecture, follow this schema
if [ "$CURR_ARCH" = "k1om" ] && [ ! -f $filename.$CURR_ARCH ]; then CURR_ARCH="mic"; fi
ARG1=$1
ARG2=$2
shift
shift
export ${@} 
if [ "x$TASKSET" != "x" ]; then 
	taskset -cp $TASKSET $$ > /dev/null 2>&1
fi
exec $filename.$CURR_ARCH $ARG1 $ARG2