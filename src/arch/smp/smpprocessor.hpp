#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "processingelement.hpp"
#include "config.hpp"
#include "smpthread.hpp"

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {

class SMPProcessor : public PE {
private:
       // config variables
       static bool useUserThreads;

	// disable copy constructor and assignment operator
	SMPProcessor(const SMPProcessor &pe);
	const SMPProcessor & operator= (const SMPProcessor &pe);
public:
	// constructors
	SMPProcessor(int id) : PE(id,&SMP) {}
	// destructor, TODO: stop all related threads
	virtual ~SMPProcessor() {}

    virtual WD & getWorkerWD () const;
    virtual WD & getMasterWD () const;
	virtual BaseThread & createThread (WorkDescriptor &wd);

	static void prepareConfig (Config &config);
};


};

#endif
