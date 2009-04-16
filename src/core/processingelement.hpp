#ifndef _NANOS_PROCESSING_ELEMENT
#define _NANOS_PROCESSING_ELEMENT

#include "queue.hpp"
#include "workdescriptor.hpp"

namespace nanos {

//TODO: define BaseThread interface
class BaseThread {
private:
int id;
bool started;
ProcessingElement *pe;

//disable copy and assigment
  BaseThread(const BaseThread &);
  const BaseThread operator= (const BaseThread &);
public:
  // constructor
  BaseThread (ProcessingElement *creator=0) : pe(creator) {}
  // destructor
  virtual ~BaseThread() {}

void run();
virtual void run_dependent () = 0;
//TODO:
// void start();
// void pause();
// void resume();
// void stop();

int getId() { return id; }

};

// forward definition
class SchedulingGroup;

class ProcessingElement {
private:
	int id;
	const Architecture *architecture;
	SchedulingGroup *schedGroup;

protected:
	virtual WorkDescriptor & getWorkerWD () const = 0;
	//TODO: priority queue? move to scheduling groups
	Queue<WD *>	readyQueue;

public:
	// constructors
	ProcessingElement(int newId,const Architecture *arch,SchedulingGroup *sg=0) :
		id(newId),architecture(arch),schedGroup(sg) {}
	// TODO: copy constructor
	ProcessingElement(const ProcessingElement &pe);
	// TODO: assignment operations
	const ProcessingElement & operator= (const ProcessingElement &pe);
	// destructor
	~ProcessingElement() {}

	/* get/put methods */
	int getId() const { return id; }
	const Architecture * getArchitecture () const { return architecture; }
	SchedulingGroup * getSchedulingGroup () const { return schedGroup; }
	void setSchedulingGroup (SchedulingGroup *sg) { schedGroup = sg; }

	virtual BaseThread & startThread (WorkDescriptor &wd) = 0;
	
	virtual void processWork () = 0;
	// TODO: if not defined, it should be a fatal exception
	virtual BaseThread & associateThisThread () = 0;
	// initializes thread-private data. Must be invoked from the thread code
	void associate();

	void startWorker();
	void submit(WD &work);

	virtual void switchTo(WD *work) = 0;
	virtual void exitTo(WD *work) = 0;
};

typedef class ProcessingElement PE;

// Each thread should be able to locate who is its PE at any moment
extern __thread PE *myPE;

};

#endif
