#include "schedule.hpp"
#include "processingelement.hpp"

using namespace nanos;

//TODO: remove, initialize policy dynamically
SchedulerPolicy * createBreadthFirstPolicy();

SchedulerPolicy * Scheduler::policy = createBreadthFirstPolicy();

void Scheduler::submit (WD &wd)
{
  PE *pe = myPE;
  // TODO: increase ready count

  WD *next = policy->atCreation(pe,wd);
  if (next) {
      pe->switchTo(next);
  }
}

void Scheduler::exit (void)
{
  PE *pe = myPE;

  // TODO:
  // Cases:
  // The WD was running on its own stack, switch to a new one
  // The WD was running on a thread stack, exit to the loop

  WD *next = policy->atExit(pe);
  if (next) {
      pe->exitTo(next);
  }
}

void Scheduler::blockOnCondition (volatile int *var, int condition)
{
	PE *pe = myPE;
	
	if ( *var != condition ) {
	    while ( *var != condition ) {
		  WD *next = policy->atBlock(pe);
		  if (next) pe->switchTo(next);
		  // TODO: implement sleeping
	    }
	}
}

void Scheduler::idle ()
{
      PE *pe = myPE;

      for ( ; ; ) {
	    WD *next = policy->atIdle(pe);
	    if (next) pe->switchTo(next);
      }
}
