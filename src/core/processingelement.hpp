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

public:

//TODO:
// void start();
// void pause();
// void resume();
// void stop();

int getId() { return id; }

};

class ProcessingElement {
private:
	int id;
	const Architecture *architecture;

protected:
	virtual WorkDescriptor & getWorkerWD () const = 0;
	//TODO: priority queue? move to scheduling groups
	Queue<WD *>	readyQueue;
	
public:
	// constructors
	ProcessingElement(int newId,const Architecture *arch) :
		id(newId),architecture(arch) {}
	// TODO: copy constructor
	ProcessingElement(const ProcessingElement &pe);
	// TODO: assignment operations
	const ProcessingElement & operator= (const ProcessingElement &pe);
	// destructor
	~ProcessingElement() {}

	/* get/put methods */
	int getId() const { return id; }
	const Architecture * getArchitecture () const { return architecture; }

	virtual BaseThread & startThread (WorkDescriptor &wd) = 0;
	
	virtual void processWork () = 0;

	void startWorker();
	void submit(WD &work);
};

typedef class ProcessingElement PE;

};

#endif
