#ifndef _NANOS_CORE_SETUP
#define _NANOS_CORE_SETUP

#include "config.hpp"

namespace nanos {

class CoreSetup {
public:
typedef enum { DEDICATED, SHARED } ExecutionMode;

private:
static int numPEs;
static bool binding;
static bool profile;
static bool instrument;
static bool verbose;
static ExecutionMode executionMode;

//added for more than one thread per pe
static int thsPerPE;;

public:

static void prepareConfig (Config &config);

static void setNumPEs (int npes) { numPEs = npes; }
static void setBinding (bool set) { binding = set; }

static int getNumPEs () { return numPEs; }
static ExecutionMode getExecutionMode () { return executionMode; }
static bool getVerbose () { return verbose; }

//set for thperper
static void setThsPerPE(int ths) { thsPerPE = ths; }
static int getThsPerPE() { return thsPerPE; }

};

};

#endif

