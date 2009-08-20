#ifndef _NANOS_SMP_WD
#define _NANOS_SMP_WD

#include <stdint.h>
#include "workdescriptor.hpp"

namespace nanos {

extern Device SMP;

class SMPDD : public DD {
public:
	typedef void (*work_fct) (void *self);

private:	
	work_fct	work;
	intptr_t *	stack;
	intptr_t *	state;
	int		stackSize; //TODO: configurable stack size

	void initStackDep (void *userf, void *data, void *cleanup);
public:
	// constructors
	SMPDD(work_fct w) : DD(&SMP),work(w),stack(0),state(0),stackSize(1024) {}
    SMPDD() : DD(&SMP),work(0),stack(0),state(0),stackSize(1024) {}
	// copy constructors
	SMPDD(const SMPDD &dd) : DD(dd), work(dd.work), stack(0), state(0) {}
	// assignment operator
	const SMPDD & operator= (const SMPDD &wd);
	// destructor
	virtual ~SMPDD() { if (stack) delete[] stack; }

	work_fct getWorkFct() const { return work; }

	bool hasStack() { return state != NULL; }
	void allocateStack();
	void initStack(void *data);
	
	intptr_t *getState() const { return state; }
	void setState (intptr_t * newState) { state = newState; }
};

inline const SMPDD & SMPDD::operator= (const SMPDD &dd)
{
  // self-assignment: ok
  if ( &dd == this ) return *this;
  
  DD::operator= (dd);
  work = dd.work;
  stack = 0;
  state = 0;
  return *this;
}

};

#endif
