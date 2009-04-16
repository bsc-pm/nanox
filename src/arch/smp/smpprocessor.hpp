#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "processingelement.hpp"
#include <pthread.h>

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {

extern Architecture SMP;

class SMPWD : public SimpleWD {
public:
	typedef void (*work_fct) (WD *self);

private:	
	work_fct	work;

public:
	// constructors
	SMPWD(work_fct w, WorkData *data=0) : SimpleWD(&SMP,data),work(w) {};
	// copy constructors
	SMPWD(const SMPWD &wd) : SimpleWD(wd), work(wd.work) {}
	// assignment operator
	const SMPWD & operator= (const SMPWD &wd);
	// destructor
	virtual ~SMPWD() {}

	work_fct getWorkFct() const { return work; }
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
	// disable copy constructor and assignment operator
	SMPProcessor(const SMPProcessor &pe);
	const SMPProcessor & operator= (const SMPProcessor &pe);
public:
	// constructors
	SMPProcessor(int id) : PE(id,&SMP) {}
	// destructor, TODO: stop all related threads
	~SMPProcessor() {}

	virtual WD & getWorkerWD () const;
	virtual BaseThread & startThread (WorkDescriptor &wd);
	virtual BaseThread & associateThisThread ();
	virtual void processWork ();
	virtual void switchTo(WD *work);
	virtual void exitTo(WD *work);

};


};

#endif
