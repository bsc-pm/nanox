#ifndef _NANOS_SMP_THREAD
#define _NANOS_SMP_THREAD

#include "smpwd.hpp"
#include <pthread.h>

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {

class SMPThread : public BaseThread {
friend class SMPProcessor;
private:
	pthread_t   pth;
    bool        useUserThreads;
    
	// disable copy constructor and assignment operator
	SMPThread(const SMPThread &th);
	const SMPThread & operator= (const SMPThread &th);

public:
	// constructor
	SMPThread(SMPWD &w, PE *pe) : BaseThread(w,pe),useUserThreads(true) {}
	// destructor
    ~SMPThread() { if (isStarted()) /*TODO: stop()*/; }

    void setUseUserThreads(bool value=true) { useUserThreads = value; }

	virtual void start();
	virtual void join();
    virtual void run_dependent (void);

    virtual void switchTo(WD *work);
    virtual void exitTo(WD *work);
};


};

#endif
