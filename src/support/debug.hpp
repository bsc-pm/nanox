#ifndef _NANOS_LIB_DEBUG
#define _NANOS_LIB_DEBUG

#include "coresetup.hpp"
#include "xstring.hpp"
#include <iostream>

namespace nanos {

//TODO: Improve information carried in Exceptions

class FatalError : public  std::runtime_error {
public:
	FatalError (const std::string &value, int peId=-1) :
		runtime_error( std::string("FATAL ERROR: [") + toString<int>(peId) + "] " + value )
		    {}

};

class FailedAssertion : public  std::runtime_error {
public:
	FailedAssertion (const std::string &value, int peId=-1) :
		runtime_error( std::string("ASSERT failed: [")+ toString<int>(peId) + "] " + value )
		    {}

};

#define fatal(msg)  throw FatalError(msg,myPE->getId());
#define fatal0(msg)  throw FatalError(msg);

#define ensure(cond) if ( !(cond) ) throw FailedAssertion(#cond, myPE->getId());
#define ensure0(cond) if ( !(cond) ) throw FailedAssertion(#cond);

#define warning(msg) { std::cerr << "WARNING: [" << myPE->getId() << "]" << msg << std::endl; }
#define warning0(msg) { std::cerr << "WARNING: [?]" << msg << std::endl; }

#define verbose(msg) \
    if (CoreSetup::getVerbose()) std::cerr << "[" << myPE->getId() << "]" << msg << std::endl;
#define verbose0(msg) \
    if (CoreSetup::getVerbose()) std::cerr << "[?]" << msg << std::endl;

};


#endif
