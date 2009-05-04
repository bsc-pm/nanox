#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "processingelement.hpp"
#include <pthread.h>

//TODO: Make smp independent from pthreads? move it to OS?

extern "C" {
void switchStacks(void *,void *,void *,void *);
}

namespace nanos {

extern Architecture SMP;

class SMPWD : public SimpleWD {
public:
	typedef void (*work_fct) (WD *self);

private:	
	work_fct	work;
	intptr_t *	stack;
	intptr_t *	state;
	int		stackSize; //TODO: configurable stack size

	void initStackDep (void *userf, void *cleanup);
public:
	// constructors
	SMPWD(work_fct w, WorkData *data=0) : SimpleWD(&SMP,data),work(w),stack(0),state(0),stackSize(1024*1024) {}
	SMPWD() : SimpleWD(&SMP,0),work(0),stack(0),state(0),stackSize(1024) {}
	// copy constructors
	SMPWD(const SMPWD &wd) : SimpleWD(wd), work(wd.work) {}
	// assignment operator
	const SMPWD & operator= (const SMPWD &wd);
	// destructor
	virtual ~SMPWD() { if (stack) delete[] stack; }

	work_fct getWorkFct() const { return work; }

	bool hasStack() { return state != NULL; }
	void allocateStack();
	void initStack();
	
	intptr_t *getState() const { return state; }
	void setState (intptr_t * newState) { state = newState; }
};

inline const SMPWD & SMPWD::operator= (const SMPWD &wd)
{
  SimpleWD::operator= (wd);
  work = wd.work;
  return *this;
}

class SMPThread : public BaseThread {
friend class SMPProcessor;
private:
	pthread_t   pth;
	SMPWD     * work;
	bool	    started;
	// disable copy constructor and assignment operator
	SMPThread(const SMPThread &th);
	const SMPThread & operator= (const SMPThread &th);

	// private constructor
	SMPThread(PE *pe) : BaseThread(pe),work(0),started(true) {}
public:
	// constructor
	SMPThread(SMPWD &w, PE *pe) : BaseThread(pe),work(&w),started(false) {}
	// destructor
	~SMPThread() { if (started) /*TODO: stop()*/; }

	// TODO: virtual?
	void start ();
	virtual void run_dependent (void);
};

class SMPProcessor : public PE {
private:
       // config variables
       static bool useUserThreads;

	// disable copy constructor and assignment operator
	SMPProcessor(const SMPProcessor &pe);
	const SMPProcessor & operator= (const SMPProcessor &pe);
public:
	// constructors
	SMPProcessor(int id, SchedulingGroup *sg=0) : PE(id,&SMP,sg) {}
	// destructor, TODO: stop all related threads
	virtual ~SMPProcessor() {}

	virtual WD & getWorkerWD () const;
	virtual BaseThread & startThread (WorkDescriptor &wd);
	virtual BaseThread & associateThisThread ();
	virtual void processWork ();
	virtual void switchTo(WD *work);
	virtual void exitTo(WD *work);

	static void prepareConfig (Config &config);

};


};

#endif
