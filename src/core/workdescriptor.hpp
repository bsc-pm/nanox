#ifndef _NANOS_WORK_DESCRIPTOR
#define _NANOS_WORK_DESCRIPTOR

#include <stdlib.h>
#include <utility>
#include <vector>
#include "workgroup.hpp"

namespace nanos {

class ProcessingElement;

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

class WorkData {
private:
	//TODO: constantize, extend to N, move to a general class?
        std::pair<int,size_t>  positions[10];
        void * data[64];
public:
	// constructor
	WorkData() {}
	// TODO:copy constructor
	WorkData(const WorkData &wd );
	// TODO: assignment operator
	const WorkData & operator= (const WorkData &wd);
	// destructor
	~WorkData() {}

	void setArguments (int totalSize, int nrefs, int nvals, ...);

	void * getValue (int narg) { return data[positions[narg].first]; }
	void * getReference (int narg) { return &data[positions[narg].first]; }

	template<typename T> T& getValue ( int narg )
	{
	    return *static_cast<T *>(getReference(narg));
	}

	template<typename T> T *getReference (int narg)
	{
	    return static_cast<T *>(getReference(narg));
	}
};

class WorkDescriptor : public WorkGroup {
private:
	WorkData *data;
protected:
	WorkData * getData () const { return data; }
public:
	// constructors
	WorkDescriptor(WorkData *wdata=0) : WorkGroup(), data(wdata) {}
	// TODO: copy constructor
	WorkDescriptor(const WorkDescriptor &wd);
	// TODO: assignment operator
	const WorkDescriptor & operator= (const WorkDescriptor &wd);
	// destructor
	virtual ~WorkDescriptor() {}
	
	virtual bool canRunIn(ProcessingElement &pe) = 0;

	void setData (WorkData *wdata) { data = wdata; }

	// TODO: change interface to WorkData
	void * getValue (int narg) { return data->getValue(narg); }
	void * getReference (int narg) { return data->getReference(narg); }

	template<typename T> T& getValue ( int narg )
	{
	    return *static_cast<T *>(data->getReference(narg));
	}

	template<typename T> T *getReference (int narg)
	{
	    return static_cast<T *>(data->getReference(narg));
	}
};


class SimpleWD : public WorkDescriptor {
private:
	// use pointers for this as is this fastest way to compare architecture
	// compatibility
	const Architecture *architecture;
public:
	// constructors
	SimpleWD(const Architecture *arch,WorkData *data=0) : WorkDescriptor(data), architecture(arch) {}
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

