#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include <iostream>

namespace nanos {

Architecture SMP ("SMP");

void * smp_bootthread (void *arg)
{
    SMPThread *self = static_cast<SMPThread *>(arg);
    
#if 0

TODO:
		cpu_set_t cpu_set;
		pid_t me = my_gettid();

		sched_getaffinity(
			(pid_t) me,
			sizeof(cpu_set),
			&cpu_set
		);
		int i;
		for (i = 0; i < CPU_SETSIZE; i++)
		{
			CPU_SET(i, &cpu_set);
		}
		sched_setaffinity(
			(pid_t) me,
			sizeof(cpu_set),
			&cpu_set);
	}

	NTH_MYSELF = (nth_desc_t*) me;
	NTH_KTH = NTH_MYSELF->vp;

	NTH_MYSELF->executed = 1;
	nth_instrument_thread_start();

	nth_data.kth_myself[NTH_KTH] = me;

	/* I'm running.. */
	nth_atm_add(&NTH_MYSELF->ndep, 1);

	/* but not for long :) */
	nth_block();


	nth_instrument_thread_end();

	pthread_exit(0);

#endif

   self->run();

    pthread_exit (0);
}

void SMPThread::start ()
{
// TODO:
//        /* initialize thread_attr: init. attr */
//        pthread_attr_init(&nth_data.thread_attr);
//        /* initialize thread_attr: stack attr */
//        rv_pthread = pthread_attr_setstack(
//                         (pthread_attr_t *) &nth_data.thread_attr,
//                         (void *) aux_desc->stack_addr,
//                         (size_t) aux_desc->stack_size
//                 );

// TODO: check return && throw exception ?
      if (pthread_create(
              &pth,
              NULL,
              smp_bootthread,
	      this )
	 )
	std::cerr << "couldn't create thread" << std::endl;
}

void SMPThread::run_dependent ()
{
    myPE->setCurrentWD(work);
    work->getWorkFct()(work);
}

void smp_run (WD *work)
{
	if (CoreSetup::getVerbose())
		std::cerr << "Starting SMP Processor on PE: "
			<< std::endl;

	
	Scheduler::idle();
}

void SMPWD::allocateStack ()
{
    stack = new intptr_t[stackSize];
}

void SMPWD::initStack ()
{
    if (!hasStack()) {
	allocateStack();
    }

    initStackDep((void *)getWorkFct(),(void *)Scheduler::exit);
}

// This is executed in between switching stacks
static void switchHelper ( intptr_t *oldState, SMPWD *oldWD, SMPWD *newWD )
{
    SimpleMessage("enqueueing task");
    oldWD->setState(oldState);
    Scheduler::queue(*oldWD);
    myPE->setCurrentWD(newWD);
}

void SMPProcessor::switchTo ( WD *wd )
{
    SMPWD *swd = static_cast<SMPWD *>(wd);
    // TODO: transform to Nth
    // TODO: swtichable work

#if 1
    if (!swd->hasStack()) {
	swd->initStack();
    }

    ::switchStacks((void *) switchHelper,
 		   (void *) currentWD,
 		   (void *) swd,
 		   (void *) swd->getState());

#else    
    (swd->getWorkFct())(wd);
     // TODO: not delete work descriptor if is a parent with pending children
    delete wd;
#endif

}

static void exitHelper ( intptr_t *oldState, SMPWD *oldWD, SMPWD *newWD )
{
    SimpleMessage("exiting task helper");
    //delete oldWD;
    oldWD->done();
    myPE->setCurrentWD(newWD);
}

void SMPProcessor::exitTo (WD *wd)
{
    SMPWD *swd = static_cast<SMPWD *>(wd);

    SimpleMessage("exiting task");
    // TODO: reuse stack

    if (!swd->hasStack()) {
	swd->initStack();
    }

    ::switchStacks((void *) exitHelper,
 		   (void *) currentWD,
 		   (void *) swd,
 		   (void *) swd->getState());    
}

void SMPProcessor::processWork ()
{
//TODO: remove
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
	SMPWD * wd = new SMPWD(smp_run,0);
	return *wd;
}

BaseThread &SMPProcessor::startThread (WorkDescriptor &helper)
{
	SMPWD &wd = static_cast<SMPWD &>(helper);
	SMPThread &th = *new SMPThread(wd,this);
	th.start();

	return th;
}

BaseThread &SMPProcessor::associateThisThread ()
{
      SMPWD *wd = new SMPWD();
      SMPThread &th = *new SMPThread(*wd,this);
      associate();
      setCurrentWD(wd);
      return th;
}


};
