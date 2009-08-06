#ifndef _NANOS_LIB_DEBUG
#define _NANOS_LIB_DEBUG

#include <stdexcept>
//Having system.hpp here generate too many circular dependences
//but it's not really needed so we can delay it most times until the actual usage
//#include "system.hpp"
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
	FailedAssertion (const char *file, const int line, const std::string &value,
	                 const std::string msg, int peId=-1) :
		runtime_error(
		       std::string("ASSERT failed: [")+ toString<int>(peId) + "] "
		               + value + ":" + msg 
		               + " (" + file + ":" + toString<int>(line)+ ")")
		    {}

};

#define fatal(msg)  throw FatalError(msg,myThread->getId());
#define fatal0(msg)  throw FatalError(msg);

#define ensure(cond,msg) if ( !(cond) ) throw FailedAssertion(__FILE__, __LINE__ , #cond, msg, myThread->getId());
#define ensure0(cond,msg) if ( !(cond) ) throw FailedAssertion(__FILE__, __LINE__, #cond, msg );

#define warning(msg) { std::cerr << "WARNING: [" << myThread->getId() << "]" << msg << std::endl; }
#define warning0(msg) { std::cerr << "WARNING: [?]" << msg << std::endl; }

#define verbose(msg) \
    if (sys.getVerbose()) std::cerr << "[" << myThread->getId() << "]" << msg << std::endl;
#define verbose0(msg) \
    if (sys.getVerbose()) std::cerr << "[?]" << msg << std::endl;

#define debug(msg) \
    if (sys.getVerbose()) std::cerr << "DBG: [" << myThread->getId() << "]" << msg << std::endl;
#define debug0(msg) \
    if (sys.getVerbose()) std::cerr << "DBG: [?]" << msg << std::endl;

};


#endif
