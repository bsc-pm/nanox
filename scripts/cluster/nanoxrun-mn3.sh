#!/bin/bash

USE_MPI="yes"
TRACE="no"
MPI_TRACE_FLAGS=""
INTERACTIVE="yes"
NUM_NODES=0
MACHINEFILE=""
QUEUE="no"
JOB_TIME="00:02"
EXTRA_NX_ARGS=""
USE_VALGRIND="no"
DISABLE_CUDA=""
TMPTRACEDIR=""
USE_WRAPPER="no"

function parseCommandLine
{
   while [ "$1" != "--" ]; do
   
      case "$1" in
     --no-interactive)
        INTERACTIVE="no"
        ;;
     --np)
        NUM_NODES=$2
        shift;
        ;;
     --machinefile)
        MACHINEFILE=$2
        shift;
        ;;
     --queue)
        QUEUE="yes"
        ;;
     --job-time)
        JOB_TIME=$2
        shift;
        ;;
     --valgrind)
        USE_VALGRIND="yes"
        ;;
     --wrap)
        USE_WRAPPER="yes"
        ;;
     --disable-cuda)
        DISABLE_CUDA="--disable-cuda"
        ;;
     *)
        if [ "$2" != "--" ] ; then
           case "$2" in
            -*)
               echo "$0: Added to NX_ARGS [$1]";
               EXTRA_NX_ARGS="$1 $EXTRA_NX_ARGS"
               ;;
            *)
               echo "$0: Added to NX_ARGS [$1 $2]";
               EXTRA_NX_ARGS="$1 $2 $EXTRA_NX_ARGS"
               shift
               ;;
           esac
        else
           echo "$0: Added to NX_ARGS [$1]";
           EXTRA_NX_ARGS="$1 $EXTRA_NX_ARGS"
        fi
        
        ;;
      esac
      shift;
   done
   
   shift
   APP=$1
   shift
   APP_ARGS=$@
}

function getModeAndLibPath
{
   NANOX_LIB=`ldd $APP | grep libnanox.so | cut -d " " -f3 | sed -e s#\/libnanox.so.*## `
   NANOX_MODE=`echo $NANOX_LIB | sed -e s#.*/##`
   ldd $APP
   echo Nanox dir: $NANOX_LIB
   echo Nanox mode: $NANOX_MODE
   echo Nodes: $NUM_NODES
}

function generateMlistAndMachinefile
{
   if [ x$INTERACTIVE = xno ] ; then
      ID=int-`date +%s`
      echo "NUM NODES : "  $NUM_NODES
   else
      if [ x$NUM_NODES = x0 ] ; then
         echo "Num nodes not set, use --np <n> flag."
         exit 1
      fi
      if [ x$MACHINEFILE = x ] ; then
         echo "Nodes file not set, use --machinefile <file> flag."
         exit 1
      fi
      
      MLIST=""
      for a in `cat $MACHINEFILE` ; do
         MLIST="$MLIST $a"
      done
      ID=int-`date +%s`
   fi
         
}

function initializeTracing
{
   if [ x$NANOX_MODE = xinstrumentation -o x$NANOX_MODE = xinstrumentation-debug ] ; then
      TRACE="yes"
   fi

   if [ x$TRACE = xyes ] ; then
      #MPI_TRACE_FLAGS="--trace"
      NANOS_TRACE_FLAGS="--instrumentation=extrae --extrae-skip-merge"

      if [ x$INTERACTIVE = xyes ] ; then
         TMPTRACEDIR=$PWD/dtmp.$APP.`printf "%04d" $NUM_NODES`.`date +%s`
         mkdir $TMPTRACEDIR
         export EXTRAE_DIR=$TMPTRACEDIR
         export EXTRAE_FINAL_DIR=$TMPTRACEDIR
      else 
         #TMPTRACEDIR=$PWD/dtmp.$APP.`printf "%04d" $NUM_NODES`.`date +%s`
         #mkdir $TMPTRACEDIR
         export TMPTRACEDIR=$TMPDIR
      fi
      export EXTRAE_DIR=$TMPTRACEDIR
      export EXTRAE_FINAL_DIR=$TMPTRACEDIR
   fi
}

function runApp
{
   if [ x$USE_MPI = xyes ] ; then
      export NX_ARGS="$EXTRA_NX_ARGS --cluster --throttle-upper 10000000 $NANOS_TRACE_FLAGS $DISABLE_CUDA"
      export NX_CLUSTER_NETWORK=ibv
      #export NX_CLUSTER_NETWORK=mpi
      export GASNET_BACKTRACE=1
      export IB_USE_GPU=1
      export CUDA_NIC_INTEROP=1
      export EXTRAE_COUNTERS=PAPI_TLB_TL,PAPI_TOT_INS,PAPI_TLB_DM
      #export GASNET_RCV_THREAD=1
      export GASNET_NETWORKDEPTH_PP=64
      export GASNET_USE_SRQ=1


      export COMPUTE_PROFILE=0
      export COMPUTE_PROFILE_CONFIG=/home/bsc15/bsc15105/profiler.cfg
      export COMPUTE_PROFILE_CSV=1

      #echo NX_CLUSTER_MEMORY is $NX_CLUSTER_MEMORY
      #echo NX_ARGS is $NX_ARGS
      #export MALLOC_CHECK_=2
      ulimit -a

      if [ x$USE_WRAPPER = xyes ]; then
         WRAPPER_CMD=/home/bsc15/bsc15105/usr/bin/nanoxwrap
      fi

      #GDB_TERM="rxvt -e gdb --args"


      if [ x$INTERACTIVE = xyes ] ; then
      #export NX_CLUSTER_NETWORK=mpi
         if [ x$USE_VALGRIND = xyes ]; then
            VALGRIND_CMD="/home/bsc15/bsc15105/usr/bin/valgrind --error-limit=no --log-file=valgrind.$APP.node%q{OMPI_COMM_WORLD_RANK}"
         fi
         FILEID=$APP.`printf "%04d" $NUM_NODES`.`date +%s`
         CMD="mpirun -x GASNET_BACKTRACE -x NX_CLUSTER_NETWORK -x NX_ARGS -x LD_LIBRARY_PATH -np $NUM_NODES $MPI_TRACE_FLAGS -machinefile $MACHINEFILE $WRAPPER_CMD $GDB_TERM $VALGRIND_CMD ./$APP $APP_ARGS"
         #> >( tee stdout.$FILEID.log ) 2> >( tee stderr.$FILEID.log )
      else
         if [ x$USE_VALGRIND = xyes ]; then
            VALGRIND_CMD="valgrind --error-limit=no --log-file=valgrind.$APP.job%q{SLURM_JOB_ID}.node%q{SLURM_PROCID}"
         fi
         CMD="mpirun -np $NUM_NODES -npernode 1  $WRAPPER_CMD $VALGRIND_CMD ./$APP $APP_ARGS"
         #export GASNET_TRACEFILE=/gpfs/scratch/bsc15/bsc15105/gasnet_traces/otpt.gasnet.%.$SLURM_JOB_ID.log
         #export GASNET_TRACEMASK=A
      fi
   else
      export SSH_SERVERS=$MLIST
      CMD="./$APP $NUM_NODES $APP_ARGS"
   fi
   echo "Executing command: $CMD"
   which mpirun
   module load openmpi
   echo "Executing command: $CMD"
   $CMD

}

function finalizeTracingNoInteractive
{
   if [ x$TRACE = xyes ] ; then
      MASTER=$(echo $LSB_MCPU_HOSTS | cut -d " " -f 1)
      ITER=0
      for i in `seq 1 $NUM_NODES ` ; do
         FILE_EXP=`printf "$TMPTRACEDIR/TRACE.??????????%06d??????.mpit" $ITER`
         FILES=`ssh $MASTER ls "$FILE_EXP"`
         echo Moving files from $i $FILE_EXP $FILES
         for k in $FILES ; do
            ssh $MASTER "echo $k on node$i >> $TMPTRACEDIR/tmp.$ID.mpits"
         done
         ITER=$(($ITER + 1))
      done
      echo Merging mpit files into a prv file...
      FILEID=trace.$APP.`printf "%04d" $NUM_NODES`.$ID
      SYM_FILE=`ssh $MASTER ls -S $TMPTRACEDIR/*.sym | head -n 1` # select the heaviest file, it contains the symbols
      ssh $MASTER mv $SYM_FILE $SYM_FILE.tmp
      ssh $MASTER rm "$TMPTRACEDIR/*.sym"
      ssh $MASTER mv $SYM_FILE.tmp $SYM_FILE
      ssh $MASTER /apps/CEPBATOOLS/extrae/latest/default/64/bin/mpi2prv -syn -f $TMPTRACEDIR/tmp.$ID.mpits -o $TMPTRACEDIR/$FILEID.prv
      scp $MASTER:$TMPTRACEDIR/$FILEID.* .
      ssh $MASTER "rm $TMPTRACEDIR/*mpit* $TMPTRACEDIR/*prv $TMPTRACEDIR/*pcf $TMPTRACEDIR/*row"
   fi
}


function finalizeTracing
{
   echo Traces in $TMPTRACEDIR  
}

function makeCmdMN3
{
   FILEID=$APP.`printf "%04d" $NUM_NODES`.`date +%s`
   NX_ARGS_FOR_FILE_NAME=$(echo $EXTRA_NX_ARGS | sed -e"s/ /_/g")
   CMDFILE=tmp.$FILEID.cmd
   if [ x$USE_VALGRIND = xyes ] ; then
      EXTRA_NX_ARGS="$EXTRA_NX_ARGS --valgrind" 
   fi
   {
      echo \#!/bin/bash
      echo \#BSUB -n $((NUM_NODES*16))
      echo \#BSUB -o otpt.$FILEID.$NX_ARGS_FOR_FILE_NAME.out
      echo \#BSUB -e otpt.$FILEID.$NX_ARGS_FOR_FILE_NAME.err
      echo \#BSUB -J $FILEID
      echo \#BSUB -R"span[ptile=16]"
      echo \#BSUB -W $JOB_TIME
      echo nanoxrun-mn3.sh --np $NUM_NODES --no-interactive $EXTRA_NX_ARGS $DISABLE_CUDA -- $APP $APP_ARGS
      echo rm $CMDFILE
   } > $CMDFILE
#cat $CMDFILE
}

########## "main" ##########
parseCommandLine $@

#if [ x$APP = xdgemm_onelevel_nomanualblocking ] && [ x$SLURM_NNODES = x32 ] ; then
# echo "Forced exit!"
# exit
#fi
   

if [ x$QUEUE = xyes ] ; then
   #generate script and submit
   makeCmdMN3
   bsub < $CMDFILE
   #rm $CMDFILE
else
   if [ x$INTERACTIVE = xyes ] ; then
      # execute
      getModeAndLibPath
      generateMlistAndMachinefile
      initializeTracing
      runApp
      if [ x$TRACE = xyes ] ; then
         finalizeTracing
      fi
   else
      getModeAndLibPath
      generateMlistAndMachinefile
      initializeTracing
      runApp
      finalizeTracingNoInteractive
   fi
fi
