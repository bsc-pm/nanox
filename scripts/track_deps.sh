#!/bin/sh
#
# Before using the command you should have the trace loaded in paraver.
# You will also need to load the configuration file displaying task numbers and find out the task
# number of the task you are interested in. This command will help you identify the incomin
# dependecences of the selected task(s).
#
# usage: track_deps.sh trace.prv list_of_task_numbers
#
# Functionality.
#   generates and loads a configuration file that shows only the specified tasks, the tasks on
#   which they depend and the dependences between them.
#   It writes to standard output the numbers of the task so that the comand can be used
#   iteratively to explore the full dependency chain. 
#

echo $*

BASEDIR=$(dirname $0)
echo $BASEDIR

export TRACE=$1
echo $#
shift
echo $#

sed "s/NB_AND_TASK_LIST/$# $*/g" $BASEDIR/task_numbers_in_path_to_selected_tasks.REF.cfg >yyy.cfg
sed "s/NB_AND_TASK_LIST/$# $*/g" $BASEDIR/tasks_in_path_to_selected.REF.cfg >xxx.cfg

wxparaver xxx.cfg
paramedir $TRACE yyy.cfg

awk ' $4 !~/^0.00/ { print $4; } ' yyy.csv | sort | awk ' BEGIN { FS = "."}; { print $1; }' >task_list.txt
echo
echo Number of tasks in path: `wc -l task_list.txt | awk '{print$1}'`
echo Task numbers: `awk ' {printf("%d ", $1); }' task_list.txt`
rm task_list.txt 

