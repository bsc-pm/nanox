#include "smpprocessor.hpp"
#include "debug.hpp"
#include <iostream>

namespace nanos {

Architecture SMP ("SMP");

void smp_run (WD *work)
{
	// recover arguments
	PE * pe = work->getValue<PE *>(0);

	if (CoreSetup::getVerbose())
		std::cerr << "Starting SMP Processor on PE: "
			<< std::endl;
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
	pe->processWork();
	pthread_exit(0);
}

void SMPProcessor::processWork ()
{
	std::string msg;
	
	SimpleMessage("processing");
	for ( ; ; ) {
	        //TODO: scheduling
		SMPWD * wd = static_cast<SMPWD *>(readyQueue.pop());
		    SimpleMessage("found work");

		    // TODO: transform to Nth
		    // TODO: swtichable work
		    (wd->getWorkFct())(wd);

		    // TODO: not delete work descriptor if is a parent with dependences
		    delete wd;
	}
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
	WorkData *data = new WorkData();
	data->setArguments(1*sizeof(void *),1,0,this);
	SMPWD * wd = new SMPWD(smp_run,data);
	return *wd;
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
              (void * (*) (void *)) work.getWorkFct(),
	      &work )
	 )
	std::cerr << "couldn't create thread" << std::endl;
}

BaseThread &SMPProcessor::startThread (WorkDescriptor &helper)
{
	SMPWD &wd = static_cast<SMPWD &>(helper);
	SMPThread &th = *new SMPThread(wd);
	th.start();

	return th;
}

/*NEXT: BaseThread &SMPProcessor::associateThread ()
{
      SMPThread &th = *new SMPThread();
      return th;
}*/


};
