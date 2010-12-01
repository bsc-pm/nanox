#!/bin/bash

PES=1
USE_MPI="yes"
TRACE="no"
MPI_TRACE_FLAGS=""
INTERACTIVE="no"
NUM_NODES=0
MACHINEFILE=""

function parseCommandLine
{
   while [ "$1" != "--" ]; do
   
      case "$1" in
     --nanoxpes)
        PES="$2"
        shift;
        ;;
     --trace)
        TRACE="yes"
        ;;
     --interactive)
        INTERACTIVE="yes"
        ;;
     --np)
        NUM_NODES=$2
        shift;
        ;;
     --machinefile)
        MACHINEFILE=$2
        shift;
        ;;
     *)
        echo "$0: Not valid argument [$1]";
        exit 1;
        ;;
      esac
   
      shift;
   done
   
   shift
   APP=$1
   shift
   APP_ARGS=$@
}

function generateMlistAndMachinefile
{
   if [ x$INTERACTIVE = xno ] ; then
      MLIST=$(/opt/perf/bin/sl_get_machine_list -u -j=\$SLURM_JOB_ID ) 
      ID=slr-$SLURM_JOB_ID

      if [ x$USE_MPI = xyes ] ; then
         MACHINEFILE="tmp.$SLURM_JOB_ID.nodes"
         for i in $MLIST ; do
            NUM_NODES=$(($NUM_NODES + 1)) ; echo $i >> $MACHINEFILE
         done
      else
         for i in $MLIST ; do
            NUM_NODES=$(($NUM_NODES + 1)) ;
         done
      fi

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
   if [ x$TRACE = xyes ] ; then
      MPI_TRACE_FLAGS="--trace"
   fi
}

function runApp
{
   if [ x$USE_MPI = xyes ] ; then
      CMD="./mpirun -np $NUM_NODES $MPI_TRACE_FLAGS --nanoxpes $PES -machinefile $MACHINEFILE ./$APP $APP_ARGS"
   else
      export SSH_SERVERS=MLIST
      CMD="./$APP -np $NUM_NODES $APP_ARGS"
   fi
   echo "Executing command: $CMD"
   ./$CMD

}

function finalizeTracing
{
   if [ x$TRACE = xyes ] ; then
      MASTER=`echo $MLIST | cut -d " " -f1 -`
      USED_MLIST=`echo $MLIST | cut -d " " -f1-$NUM_NODES -`
      ITER=0
      for i in $USED_MLIST ; do
         FILE_EXP=`printf "/scratch/TRACE.??????????%06d??????.mpit" $ITER`
         for j in $FILE_EXP ; do
            FILES=`ssh $MASTER ls $j`
            for k in $FILES ; do
               ssh $MASTER "echo $k on $i >> /scratch/tmp.$ID.mpits"
            done
         done
      ITER=$(($ITER + 1))
      done
      ssh $MASTER cat /scratch/tmp.$ID.mpits
      echo Merging mpit files into a prv file...
      ssh $MASTER ls -lh "/scratch/*.mpit"
      ssh $MASTER $HOME/extrae_install/bin/mpi2prv -f /scratch/tmp.$ID.mpits -o /scratch/$APP.$ID.prv
      scp $MASTER:/scratch/$APP.$ID.* .
      ssh $MASTER "rm /scratch/*mpit* /scratch/*prv /scratch/*pcf /scratch/*row"
   fi
}

function cleanUp
{
   if [ x$INTERACTIVE = xno ] ; then
      rm -f $MACHINEFILE
   fi
}

parseCommandLine $@
generateMlistAndMachinefile
initializeTracing
runApp
finalizeTracing
cleanUp
