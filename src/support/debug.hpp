#ifndef _NANOS_LIB_DEBUG
#define _NANOS_LIB_DEBUG

#include "coresetup.hpp"
#include <iostream>
#include <string>
#include "assert.h"

namespace nanos {

inline void SimpleMessage (const char *msg) 
{
    if (CoreSetup::getVerbose()) std::cerr << msg << std::endl;
}

inline void SimpleMessage (const std::string &msg)
{
    if (CoreSetup::getVerbose()) std::cerr << msg << std::endl;
}

#define verbose(msg) \
    if (CoreSetup::getVerbose()) std::cerr << "[" << myPE->getId() << "]" << msg << std::endl;


};


#endif
