#!/usr/bin/python
import os

def cross(*args):
	ans = [[]]
	for arg in args:
		ans = [x+[y] for x in ans for y in arg]
	return ans

def cpus(max_cpus):
	ans=[]
	for i in range(1,max_cpus+1):
		ans = ans + ['--pes='+str(i)]
	return ans

import sys
if '--help' in sys.argv:
	print 'Envorionment variables that affect this script:'
	print '    NX_TEST_MODE=\'small\'|\'medium\'|\'large\''
	print '    NX_TEST_MAX_CPUS=#CPUS'
	print '    NX_TEST_MANDATORY_ARGS=\'--nx-flag --nx-arg=val ...\''
        sys.exit()

test_mode=os.environ.get('NX_TEST_MODE')
if test_mode == None:
	test_mode='small'

max_cpus=os.environ.get('NX_TEST_MAX_CPUS')
if ( max_cpus == None ):
	max_cpus=2

mandatory_args=os.environ.get('NX_TEST_MANDATORY_ARGS')
if mandatory_args == None:
	mandatory_args=''

max_cpus=int(max_cpus)

scheduling_small=['--schedule=bf','--schedule=wf','--schedule=dbf','--shcedule=cilk']
scheduling_full=['--schedule=bf --bf-stack','--schedule=bf --no-bf-stack', '--schedule=wf --steal-parent --wf-local-policy=FIFO --wf-steal-policy=FIFO','--schedule=wf --steal-parent --wf-local-policy=FIFO --wf-steal-policy=LIFO','--schedule=wf --steal-parent --wf-local-policy=LIFO --wf-steal-policy=FIFO','--schedule=wf --steal-parent --wf-local-policy=LIFO --wf-steal-policy=LIFO','--schedule=wf --no-steal-parent --wf-local-policy=FIFO --wf-steal-policy=FIFO', '--schedule=wf --no-steal-parent --wf-local-policy=FIFO --wf-steal-policy=LIFO','--schedule=wf --no-steal-parent --wf-local-policy=LIFO --wf-steal-policy=FIFO','--schedule=wf --no-steal-parent --wf-local-policy=LIFO --wf-steal-policy=LIFO','--schedule=dbf','--shcedule=cilk']
throttle=['--throttle=dummy','--throttle=idlethreads','--throttle=numtasks','--throttle=readytasks','--throttle=taskdepth']
#barriers=['--barrier=centralized','--barrier=tree','--barrier=dissemination']
barriers=['--barrier=centralized','--barrier=tree']
others=[cpus(max_cpus),['--disable-binding','--no-disable-binding']]

if test_mode == 'small':
	configs=cross(*others+[scheduling_small])
elif test_mode == 'medium':
	configs=cross(*others+[scheduling_small]+[throttle]+[barriers])
elif test_mode == 'large':
	configs=cross(*others+[scheduling_full]+[throttle]+[barriers])

config_lines=[]
versions=''
i=1
for c in configs:
	line = 'test_ENV_ver'+str(i)+'=\"NX_ARGS=\''
	versions+='ver'+str(i)+' '
	line = line + mandatory_args
	for entry in c:
		line = line + ' ' +entry
	line = line + '\'\"'
	config_lines += [line]
	i+=1

print 'exec_versions=\"'+ versions +'\"'
for line in config_lines:
	print line

