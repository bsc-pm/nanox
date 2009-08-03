#ifndef _NANOS_WORK_DESCRIPTOR
#define _NANOS_WORK_DESCRIPTOR

#include <stdlib.h>
#include <utility>
#include <vector>
#include "workgroup.hpp"


namespace nanos {

// forward declarations
class BaseThread;
class ProcessingElement;
class WDDeque;

class Architecture
{
private:
	const char *name;
public:
	// constructor
	Architecture(const char *n) : name(n) {}
	// copy constructor
	Architecture(const Architecture &arch) : name(arch.name) {}
	// assignment operator
	const Architecture & operator= (const Architecture &arch) { name = arch.name; return *this; }
	// destructor
	~Architecture() {};

	bool operator== (const Architecture &arch) { return arch.name == name; }
};

class WorkDescriptor : public WorkGroup {
private:
	void    *data;
	bool	  tie;
	BaseThread *  tie_to;
	bool      idle;

	//Added parent for cilk scheduler: first steal parent task, next other tasks
	WorkDescriptor * parent;

	//Added reference to queue to allow dequeuing from third party (e.g. cilk scheduler)
	WDDeque * myQueue;


public:
	// constructors
	WorkDescriptor(void *wdata=0) : WorkGroup(), data(wdata), tie(false), tie_to(0), idle(false), myQueue(NULL) {}
	// TODO: copy constructor
	WorkDescriptor(const WorkDescriptor &wd);
	// TODO: assignment operator
	const WorkDescriptor & operator= (const WorkDescriptor &wd);
	// destructor
	virtual ~WorkDescriptor() {}

	WorkDescriptor * getParent() { return parent;}
	void setParent(WorkDescriptor * p) {parent = p;}

	WDDeque * getMyQueue() {return myQueue;}
	void setMyQueue(WDDeque * myQ) {myQueue = myQ;}
	bool isEnqueued() {return (myQueue != NULL);}


	/* named arguments idiom */
	WorkDescriptor & tied () { tie = true; return *this; }
	WorkDescriptor & tieTo (BaseThread &pe) { tie_to = &pe; tie=false; return *this; }

	bool isTied() const { return tie_to != NULL; }
	BaseThread * isTiedTo() const { return tie_to; }
	
	virtual bool canRunIn(ProcessingElement &pe) = 0;

	void setData (void *wdata) { data = wdata; }
    void * getData () const { return data; }

	bool isIdle () const { return idle; }
	void setIdle(bool state=true) { idle = state; }
};


class SimpleWD : public WorkDescriptor {
private:
	// use pointers for this as is this fastest way to compare architecture
	// compatibility
	const Architecture *architecture;
public:
	// constructors
	SimpleWD(const Architecture *arch,void *data=0) : WorkDescriptor(data), architecture(arch) {}
	// copy constructor
	SimpleWD(const SimpleWD &wd) : WorkDescriptor(wd), architecture(wd.architecture)  {}
	// assignment operator
	const SimpleWD & operator= (const SimpleWD &wd);
	// destructor
	~SimpleWD() {}

	virtual bool canRunIn(ProcessingElement &pe);
};

inline const SimpleWD & SimpleWD::operator= (const SimpleWD &wd)
{
  // self-assignment: ok
  WorkDescriptor::operator=(wd);
  architecture = wd.architecture;
  return *this;
}

// class MultiArchWD : public WorkDescriptor {
// private:
// public:
//   
// };

typedef class WorkDescriptor WD;

};

#endif

