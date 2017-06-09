#!/usr/bin/env python2
import os

def cross(*args):
	ans = [[]]
	for arg in args:
      # Make sure this argument is not empty
		if arg:
		   ans = [x+[y] for x in ans for y in arg]
	return ans

def cpus(max_cpus):
	ans=[]
	for i in range(1,max_cpus+1):
		ans = ans + ['--smp-workers='+str(i)]
	return ans

test_mode = os.environ.get('NX_TEST_MODE', 'small') or 'small'
max_cpus = os.environ.get('NX_TEST_MAX_CPUS', 2) or 2
test_schedule = os.environ.get('NX_TEST_SCHEDULE')
test_architecture = os.environ.get('NX_TEST_ARCH')

# Process program arguments (priority to env vars)
from optparse import OptionParser
import sys

header ='Nanox config generator 0.1\n\n'+\
	'Envorionment variables that affect this script:\n'+\
	'   NX_TEST_MODE=\'performance\'|\'small\'|\'medium\'|\'large\'   -  \'small\' by default\n'+\
	'   NX_TEST_MAX_CPUS=#CPUS                  -  2 by default\n'+\
	'   NX_TEST_SCHEDULE=[scheduler]\n'+\
	'   NX_TEST_ARCH=[architecture]\n'
if '-h' in sys.argv or '--help' in sys.argv:
	print header

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-a", metavar="\"a1|a2,b1|b2,..\"", dest="additional",
                  help="Comma separated lists of aditional options ('|' separates incompatible alternatives ) combined in the configurations generated")
parser.add_option("-m", choices=['performance','small','medium','large'], dest="mode",
                  help="Determines the number of execution versions for each test combining different runtime options.")
parser.add_option("-c","--cpus", metavar="n", type='int', dest="cpus",
                  help="Each configuration will be tested for 1 to n CPUS")
parser.add_option("-d", "--deps", metavar="\"a1,b1,..\"", dest="deps_plugins",
                  help="Comma separated lists of dependencies plugins combined in the configurations generated")

(options, args) = parser.parse_args()

if len(args) != 0:
	parser.error("Wrong arguments")

addlist=[]
if options.additional:
	additional=options.additional
	additional=additional.split(',')
	for a in additional:
		addlist=addlist+[a.split('|')]
if options.mode:
	test_mode=options.mode
if options.cpus:
	max_cpus=options.cpus
depslist = []
if options.deps_plugins:
	deps_plugins=options.deps_plugins
	deps_plugins=deps_plugins.split(',')
	for d in deps_plugins:
		depslist=depslist+["--deps="+d]

max_cpus=int(max_cpus)

scheduling_performance=[]
scheduling_small=['--schedule=dbf','--schedule=dbf --schedule-priority']
scheduling_large=['--schedule=bf --bf-stack','--schedule=bf --no-bf-stack','--schedule=dbf', '--schedule=affinity']
throttle=['--throttle=dummy','--throttle=idlethreads','--throttle=numtasks','--throttle=readytasks','--throttle=taskdepth']
barriers=['--barrier=centralized','--barrier=tree']
binding=['--disable-binding','--no-disable-binding']
architecture=['--architecture=smp']

if test_schedule is not None:
   scheduling_performance=['--schedule='+test_schedule]
   scheduling_small=['--schedule='+test_schedule]
   scheduling_large=['--schedule='+test_schedule]

if test_architecture is not None:
   architecture=['--architecture='+test_architecture]

if test_mode == 'performance':
	configs=cross(*[cpus(max_cpus)]+[scheduling_performance]+addlist)
elif test_mode == 'small':
	configs=cross(*[cpus(max_cpus)]+[binding]+[architecture]+[scheduling_small]+[depslist]+addlist)
elif test_mode == 'medium':
	configs=cross(*[cpus(max_cpus)]+[binding]+[architecture]+[scheduling_small]+[throttle]+[barriers]+[depslist]+addlist)
elif test_mode == 'large':
	configs=cross(*[cpus(max_cpus)]+[binding]+[architecture]+[scheduling_large]+[throttle]+[barriers]+[depslist]+addlist)

config_lines=[]
versions=''
i=1
for c in configs:
	line = 'test_ENV_ver'+str(i)+'=\"NX_ARGS=\$NX_ARGS\' '
	versions+='ver'+str(i)+' '
	for entry in c:
		line = line + ' ' +entry
	line = line + '\'\"'
	config_lines += [line]
	i+=1

print 'exec_versions=\"'+ versions +'\"'
for line in config_lines:
	print line

