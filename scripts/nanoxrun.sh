#!/bin/bash

PES=1
USE_MPI="yes"
TRACE="no"
MPI_TRACE_FLAGS=""
INTERACTIVE="yes"
NUM_NODES=0
MACHINEFILE=""
QUEUE="no"

function parseCommandLine
{
   while [ "$1" != "--" ]; do
   
      case "$1" in
     --nanoxpes)
        PES="$2"
        shift;
        ;;
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

function getModeAndLibPath
{
   NANOX_LIB=`ldd $APP | grep libnanox.so | cut -d " " -f3 | sed -e s#\/libnanox.so.*## `
   NANOX_MODE=`echo $NANOX_LIB | sed -e s#.*/##`
   ldd $APP
   echo Nanox dir: $NANOX_LIB
   echo Nanox mode: $NANOX_MODE
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
   if [ x$NANOX_MODE = xinstrumentation ] ; then
      TRACE="yes"
   fi

   if [ x$TRACE = xyes ] ; then
      MPI_TRACE_FLAGS="--trace"
      NANOS_TRACE_FLAGS="--instrumentation=extrae"
      export TMPDIR=/scratch
      export EXTRAE_DIR=/scratch
      export EXTRAE_FINAL_DIR=/scratch
   fi
}

function runApp
{
   if [ x$USE_MPI = xyes ] ; then
      if [ x$INTERACTIVE = xyes ] ; then
         CMD="./mpirun -np $NUM_NODES $MPI_TRACE_FLAGS --nanoxpes $PES -machinefile $MACHINEFILE ./$APP $APP_ARGS"
      else
         export LD_LIBRARY_PATH=$NANOX_LIB:/gpfs/apps/GCC/4.4.0/lib64:/opt/osshpc/mpich-mx/64/lib/shared:/gpfs/apps/GCC/4.4.0/lib:/opt/osshpc/mpich-mx/32/lib/shared
         export NX_ARGS="--pes $PES $NANOS_TRACE_FLAGS"
         CMD="srun ./$APP $APP_ARGS"
      fi
   else
      export SSH_SERVERS=$MLIST
      CMD="./$APP $NUM_NODES $APP_ARGS"
   fi
   echo "Executing command: $CMD"
   $CMD

}

function finalizeTracing
{
   if [ x$TRACE = xyes ] ; then
      MASTER=`echo $MLIST | cut -d " " -f1 -`
      USED_MLIST=`echo $MLIST | cut -d " " -f1-$NUM_NODES -`
      ITER=0
      for i in $USED_MLIST ; do
         FILE_EXP=`printf "/scratch/TRACE.??????????%06d??????.mpit" $ITER`
         FILES=`ssh $MASTER ls $FILE_EXP`
         for k in $FILES ; do
            ssh $MASTER "echo $k on $i >> /scratch/tmp.$ID.mpits"
         done
      ITER=$(($ITER + 1))
      done
      ssh $MASTER cat /scratch/tmp.$ID.mpits
      echo Merging mpit files into a prv file...
      FILEID=$APP.`printf "%04d" $NUM_NODES`.$PES.$ID
      ssh $MASTER ls -lh "/scratch/*.mpit"
      PCF_FILE=`ssh $MASTER ls /scratch/*.pcf`
      scp $MASTER:$PCF_FILE $FILEID.pcf
      ssh $MASTER $HOME/extrae_install/bin/mpi2prv -f /scratch/tmp.$ID.mpits -o /scratch/$FILEID.prv
      ssh $MASTER rm /scratch/$FILEID.pcf
      scp $MASTER:/scratch/$FILEID.* .
      ssh $MASTER "rm /scratch/*mpit* /scratch/*prv /scratch/*pcf /scratch/*row"
   fi
}

function cleanUp
{
   if [ x$INTERACTIVE = xno ] ; then
      rm -f $MACHINEFILE
   fi
}

function makeCmd
{
   FILEID=$APP.`printf "%04d" $NUM_NODES`.$PES.`date +%s`
   CMDFILE=tmp.$FILEID.cmd
   {
      echo \#!/bin/bash
      echo \#@ job_name = $FILEID
      echo \#@ class = bsc_cs
      echo \#@ initialdir = .
      echo \#@ output = $FILEID.%j.out
      echo \#@ error= $FILEID.%j.err
      echo \#@ total_tasks = $NUM_NODES
      echo \#@ cpus_per_task = 4
      echo \#@ nodeset= clos
      echo \#@ wall_clock_limit = 00:45:00
      echo \#@ tracing = 1
      echo ""
      echo ./nanoxrun.sh --no-interactive --nanoxpes $PES -- $APP $APP_ARGS
   } > $CMDFILE
#cat $CMDFILE
}

########## "main" ##########
parseCommandLine $@

if [ x$QUEUE = xyes ] ; then
   #generate script and submit
   makeCmd
   mnsubmit $CMDFILE
   rm $CMDFILE
else
   # execute
   getModeAndLibPath
   generateMlistAndMachinefile
   initializeTracing
   runApp
   finalizeTracing
   cleanUp
fi
