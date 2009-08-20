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

class Device
{
private:
	const char *name;
public:
	// constructor
	Device(const char *n) : name(n) {}
	// copy constructor
	Device(const Device &arch) : name(arch.name) {}
	// assignment operator
	const Device & operator= (const Device &arch) { name = arch.name; return *this; }
	// destructor
	~Device() {};

	bool operator== (const Device &arch) { return arch.name == name; }
};

// This class holds the specific data for a given device
class DeviceData
{
private:
    // use pointers for this as is this fastest way to compare architecture
    // compatibility
    const Device *architecture;
public:
    // constructors
    DeviceData(const Device *arch) : architecture(arch) {}
    // copy constructor
    DeviceData(const DeviceData &dd) : architecture(dd.architecture)  {}
    // assignment operator
    const DeviceData & operator= (const DeviceData &wd);

    bool isCompatible(const Device &arch) { return architecture == &arch; }
    
    // destructor
    virtual ~DeviceData() {}
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

    // Supported devices for this workdescriptor
    int         num_devices;
    DeviceData **devices;
    DeviceData *active_device;
    
public:
	// constructors
	WorkDescriptor(int ndevices, DeviceData **devs,void *wdata=0) :
           WorkGroup(), data(wdata), tie(false), tie_to(0), idle(false),
		   parent(NULL), myQueue(NULL), num_devices(ndevices), devices(devs), active_device(0) {}
    WorkDescriptor(DeviceData *device,void *wdata=0) :
           WorkGroup(), data(wdata), tie(false), tie_to(0), idle(false),
           parent(NULL), myQueue(NULL), num_devices(1), devices(0), active_device(device) {}
	// TODO: copy constructor
	WorkDescriptor(const WorkDescriptor &wd);
	// TODO: assignment operator
	const WorkDescriptor & operator= (const WorkDescriptor &wd);
	// destructor
	virtual ~WorkDescriptor() { /*TODO*/ }

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

	void setData (void *wdata) { data = wdata; }
    void * getData () const { return data; }

	bool isIdle () const { return idle; }
	void setIdle(bool state=true) { idle = state; }

    /* device related methods */
    DeviceData * findDeviceData (const Device &device) const;
    bool canRunIn (const Device &device) const;
    bool canRunIn (const ProcessingElement &pe) const;
    DeviceData & activateDevice (const Device &device);
    DeviceData & getActiveDevice () const { return *active_device; }
    bool hasActiveDevice() const { return active_device != NULL; }

};

inline const DeviceData & DeviceData::operator= (const DeviceData &dd)
{
  // self-assignment: ok
  architecture = dd.architecture;
  return *this;
}

typedef class WorkDescriptor WD;
typedef class DeviceData DD;

};

#endif

