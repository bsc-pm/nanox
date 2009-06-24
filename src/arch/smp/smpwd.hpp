#ifndef _NANOS_SMP_WD
#define _NANOS_SMP_WD

#include <stdint.h>

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
	SMPWD(work_fct w, WorkData *data=0) : SimpleWD(&SMP,data),work(w),stack(0),state(0),stackSize(1024) {}
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

};

#endif
