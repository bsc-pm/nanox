#!/bin/bash

VERSION=0.6.0
PES=1
USE_MPI="yes"
TRACE="no"
MPI_TRACE="(disabled)" # This value MUST be different from 'disabled' to detect the user preference
NUM_NODES=0
TPN=1 # Tasks per node, by default a single MPI process (or task) per node
MACHINEFILE=""
KEEP="no"
KEEPFILE=
RMKEEP=borrakeep.sh
QUEUE="yes"
JOB_TIME="00:20:00"
SUBMIT="yes"
XTRA_NX_ARGS=--force-tied-master
if [ x$NX_GPUS == x ]; then 
	export NX_GPUS=0
fi
if [ x$NX_GPUS == x0 ]; then
	USE_CUDA="disabled"
	CUDAFLAG="--disable-cuda=yes"
else
	USE_CUDA="$NX_GPUS"
	CUDAFLAG="--gpus=$NX_GPUS"
fi

function help
{
	SC=`basename $0`
	echo "Syntax: $SC [OPTIONS] -- APP [APP_ARGS]"
	echo ""
	echo "OPTIONS:"
	echo "  --np x          : Number of nodes to use [Default = $NUM_NODES]"
	echo "  --pes x "
	echo "  --nanoxpes x    : Total number of processors per node to use [Default = $PES]"
	echo "  --gpus x        : Use x GPUS (CUDA runtime) [Default = $USE_CUDA]"
	echo "  --mpitrace"
	echo "  --no-mpitrace   : Enable/Disable MPI tracing [Default = $MPI_TRACE]"
	echo "  --machinefile x : Use nodes from file 'x' (a node name per line) "
	echo "  --queue         : Generate a script and submit it to queue system [Default = $QUEUE]"
	echo "    --job-time x  : Limit execution time to 'x' [Default = $JOB_TIME]"
	echo "  --slurm x       : Use specific queue header for : [MACHINE = $MACHINE]"
	echo "          MN      : Marenostrum (MOAB)"
	echo "          MT      : Minotauro (SLURM)"
	echo "          tibidabo: Tibidabo (SLURM)"
	echo "  --keep-files    : Keep generated files [Default = $KEEP]"
	echo "                    Use ./$RMKEEP to delete ALL kept files"
	echo "                    $RMKEEP is automatically generated."
	echo "  -N              : Do not submit application, but keep the script"
	echo "  --exclusive     : Launch the application using the whole node (to avoid other jobs in the same node)"
	echo "  --version       : Show version number"
	echo ""
	echo " Following environment variables are honored:"
	echo "      EXTRAE_HOME=$EXTRAE_HOME"
	echo "      EXTRAE_CONFIG_FILE=$EXTRAE_CONFIG_FILE"
	echo "      EXTRAE_LABELS=$EXTRAE_LABELS"
	echo "      EXTRAE_FINAL_DIR=$EXTRAE_FINAL_DIR"
	echo "      NX_GPUS=$NX_GPUS"
	echo "      MACHINE=$MACHINE"
	exit -1
}

function parseCommandLine
{
	while [ "$1" != "--" ]; do

		case "$1" in
			--nanoxpes|--pes)
				PES="$2"
				OMP_NUM_THREADS=$PES
				shift;
				;;
			--np)
				NUM_NODES=$2
				shift;
				;;
			--machinefile)
				MACHINEFILE=$2
				QUEUE="no"
				shift;
				;;
			--no-mpitrace)
				MPI_TRACE="disabled"
				;;
			--mpitrace)
				MPI_TRACE="enabled"
				;;
			--keep-files)
				KEEP="yes"
				;;
			--gpus)
				CUDAFLAG="--gpus=$2"
				shift
				;;
			--no-queue|--interactive)
				QUEUE="no"
				;;
			--queue)
				QUEUE="yes"
				;;
			--slurm)
				MACHINE=$2
				shift;
				;;
			--job-time)
				JOB_TIME=$2
				shift;
				;;
			-N)
				SUBMIT="no"
				KEEP="yes"
				;;
			--exclusive)
				EXCLUSIVE="yes"
				;;
			--help|-h)
				help
				;;
			--version)
				echo "`basename $0` v$VERSION"
				exit
				;;
			*)
				echo "$0: Not valid argument [$1]";
				help
				;;
		esac

		shift;
	done

	shift
	APP=$1
	shift
	APP_ARGS=$@
}

function delete
{
	FILE=$@
	echo "$FILE" >> $KEEPFILE
}

function deleteallfiles
{
	echo echo "Removing $1"
	echo for i in \`cat $1\` 
    echo do
	echo '  if [ -f $i ]; then'
	echo '	rm $i'
	echo '  fi'
	echo done
	echo rm $1
}


function getModeAndLibPath
{
	NANOX_MODE="no"
	NANOX_LIB=`ldd $APP | grep libnanox.so | cut -d " " -f3 `
	if [ $NANOX_LIB ]; then
		NANOX_LIB=`dirname $NANOX_LIB`
		NANOX_MODE=`basename $NANOX_LIB`
		echo Nanox dir: $NANOX_LIB
		echo Nanox mode: $NANOX_MODE
	else
		## In the MPI case, use OMP_NUM_THREADS to define the number of tasks per node
		if [ "x$OMP_NUM_THREADS" != "x" ]; then
			TPN=$OMP_NUM_THREADS
			PES=$TPN
		fi
	fi
}

function generateMlistAndMachinefile
{
	if [ x$QUEUE = xyes ] ; then
		if [ x$MACHINE == xtibidabo ]; then
			echo 'MLIST=$(/opt/perf/bin/scontrol show hostnames $SLURM_NODELIST)'
		fi
		if [ x$MACHINE == xMN ] ; then
			MACHINEFILE="${FILEID}.\$SLURM_JOB_ID.nodes"
			echo "MACHINEFILE=$MACHINEFILE"
			echo 'MLIST=$(paste -s $MACHINEFILE)'
			delete $MACHINEFILE
		fi
		if [ x$MACHINE == xjudge ] ; then
			MACHINEFILE=\${PBS_NODEFILE}
			echo "MACHINEFILE=$MACHINEFILE"
			echo 'MLIST=$(paste -s $MACHINEFILE)'
			## Due to the use of ppn=2, we need to reorganize this file.
			## Otherwise, two mpi processes will go to the same node.
			## Original: aabbcc
			## Result  : abcabc
			#echo "uniq \$PBS_NODEFILE > $MACHINEFILE"
			#echo "uniq \$PBS_NODEFILE >> $MACHINEFILE"
		fi
		if [ x$MACHINE == xMT ]; then
			echo 'MLIST=$(sl_get_machine_list -u -j=$SLURM_JOBID)'
		fi
	fi
	showMlist
}

function showMlist
{
    echo 'echo "Executing at: ${MLIST}"'
}

function initializeTracing
{
	if [ x$NANOX_MODE = xinstrumentation -o x$NANOX_MODE = xinstrumentation-debug ] ; then
		TRACE="yes"
		# Automatically *enable* MPI tracing unless 'user' states otherwise
		[ x$MPI_TRACE != xdisabled ] &&  MPI_TRACE="enabled"
	fi

	#   export TMPDIR=$PWD
	if [ x$TRACE = xyes ] ; then
		NANOS_TRACE_FLAGS="$XTRA_NX_ARGS --instrumentation=extrae "
		if [ x$MPI_TRACE == xenabled ]; then
			NANOS_TRACE_FLAGS="$NANOS_TRACE_FLAGS --extrae-skip-merge --extrae-keep-mpits"
			#NANOS_TRACE_FLAGS="$NANOS_TRACE_FLAGS --extrae-disable-init" # versio Minotauro experimental per tutorial MontBlanc (To be DELETED)
			#NANOS_TRACE_FLAGS="$NANOS_TRACE_FLAGS --extrae-skip-init --extrae-skip-fini" # versio final 0.7 --> Ja no es necessari amb extrae 2.3
		fi
	fi
	if [ x$MPI_TRACE == xenabled ]; then
		if [ ! -f $EXTRAE_HOME/lib/libnanosmpitrace.so ]; then
			echo "ERROR: Tracing library libnanosmpitrace not found at EXTRAE_HOME/lib!"
			exit 1
		fi
		LD_PRELOAD="\$EXTRAE_HOME/lib/libnanosmpitrace.so:\$EXTRAE_HOME/lib/libnanosmpitracef.so"
		if [ x$EXTRAE_CONFIG_FILE == x ]; then
			EXTRAE_ON=1
		fi
	fi
}

function runApp
{
	[ "x$NANOX_MODE" != "xno" ] && NX_ARGS="--pes \$OMP_NUM_THREADS $CUDAFLAG "
	if [ x$MACHINE == xjudge ]; then
		ENVSH=./$CMDFILE.sh
		delete $ENVSH
		script_header      >  $ENVSH
		makeEnv            >> $ENVSH
	    echo '$*'          >> $ENVSH
		chmod u+x $ENVSH
	else
		makeEnv
	fi
	
	CMD=
	if [ x$QUEUE = xno ] ; then
		# mpirun NO exporta variables d'entorn i per tant cal fer servir un script que ho faci
		#CMD="mpiexec -np $NUM_NODES -machinefile $MACHINEFILE ./$APP $APP_ARGS"
		CMD="mpirun -np $NUM_NODES -machinefile $MACHINEFILE $ENVSH  \$TRACESH"
	else
		CMD="srun $ENVSH  \$TRACESH"
		if [ x$MACHINE == xjudge ]; then
			CMD="mpiexec -np $NUM_NODES $ENVSH \$TRACESH"
		fi
	fi

	#echo "Executing command: $CMD ./$APP $APP_ARGS"
	# Check if the filename has a slash, so if it does not it assumes it is a local filename
	# (needed for tibidbado)
	X=`echo $APP | awk -F/ '{print NF}'`
	if [ $X == 1 ]; then
		APP=./$APP
	fi
	echo $CMD $APP $APP_ARGS
}

function finalizeTracing
{
		echo "##################################################"
		echo unset LD_PRELOAD
		echo 'if [[ ( $SLURM_NODEID -eq 0 ) && ( $SLURM_LOCALID -eq 0 ) ]]; then'
		echo MPITSF=TRACE.mpits
		echo SET_0=./set-0
		echo '# _find_ command returns the whole path and so, first '
		echo '# enter into the directory to get relative directories'
		echo '# and then the linking process just needs to add the ../../'
		echo 'pushd $EXTRAE_FINAL_DIR'
		echo 'MPITSF=(`ls -S trace??????/TRACE.mpits`)'
		echo 'MPITSF=${MPITSF[0]}'
		echo 'if [ x$MPITSF != x ]; then'
		echo '  if [ -f  $MPITSF ]; then'
		echo '    MPITSD=`dirname $MPITSF`'
		echo '    SET_0=$MPITSD/set-0'
		echo '# TRACE.mpits and *.mpit are in different directorys, '
		echo '# group them to a common one (EXTRAE_FINAL_DIR/trace/set-0)'
		echo '    for i in trace??????/set-0/*.mpit; do'
		echo '          ln -s ../../$i $SET_0 >& /dev/null'
		echo '    done'
		echo '# extrae 2.3 generates .sym files with function symbols for each node.'
		echo '# Aggregate all generated symbol files in a single TRACE.sym'
		echo '    for i in trace??????/set-0/*.sym ; do'
		echo '        grep -h "^O" $i >> kk.sym'
		echo '    done'
		echo '    sed -i /^O/d ${MPITSD}/TRACE.sym'
		echo '    sort kk.sym |uniq >> ${MPITSD}/TRACE.sym'
		echo '    rm kk.sym'
		echo '  else'
		echo '    # EXTRAE_FINAL_DIR is not used in the XML control file. Guessing...'
		echo '    # XML contains current directory of final-directory not enabled'
		echo '    MPITSF=TRACE.mpits'
		echo '  fi'
		# Si EXTRAE_LABELS esta definit, overwrite!
		if [ "x${EXTRAE_LABELS}" != "x" ] ; then
		echo '  LABELS="EXTRAE_LABELS='${EXTRAE_LABELS}'"'
		else
		echo '  PCF_FILE=(`ls trace??????/*.pcf`)'
		echo '  [ x${PCF_FILE} != x ] && LABELS="EXTRAE_LABELS=$EXTRAE_FINAL_DIR/${PCF_FILE[0]}"'
		fi
		echo ' popd'
		echo    PRV=$FILEID.${TMSTMP}.prv
		echo '  echo   Merging mpit files into a prv file...[$PRV]'
		echo '  X=" $LABELS $EXTRAE_HOME/bin/mpi2prv -f $EXTRAE_FINAL_DIR/$MPITSF -e '$APP' -o $EXTRAE_FINAL_DIR/$PRV"'
		echo '  echo $X'
		echo '  $X'
		echo 'else'
		echo '  echo "ERROR: I do not know how to merge the tracefile :("'
		echo 'fi'
		echo 'fi'
}

function cleanUp
{
	if [ x$KEEP == xno ]; then
		deleteallfiles $KEEPFILE
	fi
}

function script_header
{
	echo \#!/bin/bash
	echo \# Automatically generated file by $0 $VERSION
}

function makeCmd
{
	script_header
	if [ x$MACHINE == xjudge ]; then
	#      JUDGE only vvvvvvvvvv
		# ppn MUST BE 1 instead of $PES because MOAB generates a wrong machinefile.
		# Ex: An execution of 3 nodes with 2 threads/node gives: AABBCC instead of ABCABC
		#    and so there are 2 mpi tasks inside the same node!!
		#    QUESTION: the nodes/processors are shared with other jobs/users???
		echo \#MSUB -l nodes=$NUM_NODES:ppn=$PES:gpus=$NX_GPUS:performance
		echo \#MSUB -l walltime=$JOB_TIME
		echo \#MSUB -v tpt=$PES
		#export OMP_NUM_THREADS=$PES # This is REALLY necessary!!
		#echo NSLOTS=$NUM_NODES
	#      JUDGE only ^^^^^^^^^^
	else
		echo \#@ job_name = $FILEID.${TMSTMP}
		echo \#@ initialdir = .
		echo \#@ output = $FILEID.${TMSTMP}.%j.out
		echo \#@ error= $FILEID.${TMSTMP}.%j.err
		echo \#@ total_tasks = $NUM_NODES
		echo \#@ wall_clock_limit = $JOB_TIME
		if [ x$MACHINE == xMT ]; then
			#		   MINOTAURO only vvvvvvvvvv
			echo \#@ tasks_per_node = $TPN
			if [ x$EXCLUSIVE == xyes ]; then
				echo \#@ gpus_per_node = 2
				# cpus_per_task = total_cpus_per_node / tasks_per_node
				CPT=$((12/$TPN))
				echo \#@ cpus_per_task = $CPT
				echo \#@ node_usage = not_shared
			else 
				echo \#@ cpus_per_task = $PES
			fi
			#		   MINOTAURO only ^^^^^^^^^^
		fi
		if [ x$MACHINE == xMN ]; then
			#		   MARENOSTRUM only vvvvvvvvvv
			echo \#@ cpus_per_task = 4
			echo \#@ nodeset= clos
			echo \#@ tracing = 1
			#		   MARENOSTRUM only ^^^^^^^^^^
			#echo \#@ class = bsc_cs
		fi
		if [ x$MACHINE == xtibidabo ]; then
			echo \#@ tasks_per_node = 1
			echo \#@ cpus_per_task = 2
			echo 'echo 10>>  /proc/self/oom_adj'
		fi
	fi
	echo ""
}

function makeEnv
{
	#   export NX_THROTTLE=idlethreads
	#   export NX_SCHEDULE=affinity
	#   export NX_CLUSTER_MEMORY=1076363264

	#   echo NX_THROTTLE is $NX_THROTTLE
	#   echo NX_SCHEDULE is $NX_SCHEDULE
	#   echo NX_CLUSTER_NODE_MEMORY is $NX_CLUSTER_NODE_MEMORY
	#   echo NX_DISABLECUDA is $NX_DISABLECUDA
	# echo export OBJECT_MODE=64
	# echo export MXMPI_RECV=blocking
		echo ""
		#      echo export LD_DEBUG=all
		[ x$OMP_NUM_THREADS != x ] && echo export OMP_NUM_THREADS=$OMP_NUM_THREADS
		[ x$LD_LIBRARY_PATH != x ]  && echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
		echo ""
		#[ x$LD_PRELOAD != x ]       && echo export LD_PRELOAD=$LD_PRELOAD
		echo "##################################################"
		[ "x$NANOX_MODE" != "xno" ] && echo export NX_ARGS="\"$NX_ARGS \""
}

function makeTrace
{
	script_header
	[ x$EXTRAE_CONFIG_FILE != x ] && echo export EXTRAE_CONFIG_FILE=$EXTRAE_CONFIG_FILE
	[ x$EXTRAE_ON != x ]          && echo export EXTRAE_ON=1
	[ x$EXTRAE_DIR != x ]         && echo export EXTRAE_DIR=$EXTRAE_DIR
	[ x$EXTRAE_FINAL_DIR != x ]   && echo export EXTRAE_FINAL_DIR=$EXTRAE_FINAL_DIR
	[ x$EXTRAE_HOME != x ]        && echo export EXTRAE_HOME=$EXTRAE_HOME
	echo export LD_PRELOAD=$LD_PRELOAD
	[ "x$NANOX_MODE" != "xno" ]   && echo export NX_ARGS="\"\$NX_ARGS $NANOS_TRACE_FLAGS\""
	[ x$EXTRAE_FINAL_DIR == x ] && EXTRAE_FINAL_DIR=.
	echo export EXTRAE_FINAL_DIR=$EXTRAE_FINAL_DIR
	echo "##################################################"
	echo '$*'
}

########## "main" ##########
parseCommandLine $@

if [ x$MACHINE == x ]; then
	echo "ERROR: --slurm or MACHINE variable not defined"
	exit 1
fi
if [ x$NUM_NODES = x0 ] ; then
	echo "Num nodes not set, use --np <n> flag."
	exit 1
fi
if [ x$QUEUE = xno -a x$MACHINEFILE = x ] ; then
	echo "Nodes file not set, use --machinefile <file> flag."
	exit 1
fi

# Generate Script to delete all keep-files
echo '#!/bin/sh' > $RMKEEP
echo 'for j in .*.keep; do ' >> $RMKEEP
deleteallfiles \$j >> $RMKEEP
echo 'done' >> $RMKEEP
echo "rm $RMKEEP" >> $RMKEEP # Auto-delete itself
chmod u+x $RMKEEP


# execute
getModeAndLibPath 

FILEID=`basename $APP`.`printf "%04d" $NUM_NODES`.$PES
TMSTMP=`date +%s`
KEEPFILE=.$FILEID.$TMSTMP.keep	#Hide these file from user...

initializeTracing 

#generate script and submit
CMDFILE=run.$FILEID.cmd
delete $CMDFILE
makeCmd > $CMDFILE

generateMlistAndMachinefile >> $CMDFILE

if [ x$TRACE = xyes -o x$MPI_TRACE = xenabled ] ; then
	TRACESH=./$CMDFILE.trace.sh
	delete $TRACESH
	makeTrace           > $TRACESH
	chmod u+x $TRACESH
	echo 'TRACING=1' >> $CMDFILE
	echo 'if [ ${TRACING} == 1 ]; then'	>>$CMDFILE
	echo '	echo "Paraver Tracing is ENABLED"'	>>$CMDFILE
	echo "	TRACESH=$TRACESH"	>>$CMDFILE
	echo 'fi'	>>$CMDFILE

fi

runApp >> $CMDFILE
if [ x$TRACE = xyes -o x$MPI_TRACE = xenabled ] ; then
	finalizeTracing >> $TRACESH
fi
cleanUp >> $CMDFILE

SUBMIT_CMD=
if [ x$QUEUE = xyes ] ; then
	SUBMIT_CMD="mnsubmit"
	if [ x$MACHINE == xjudge ]; then
		SUBMIT_CMD="msub -V"
	fi
fi
SUBMIT_CMD="$SUBMIT_CMD ./$CMDFILE"

echo $SUBMIT_CMD
[ x$SUBMIT == xyes ] && $SUBMIT_CMD

