#ifndef _NANOS_SYSTEM
#define _NANOS_SYSTEM

#include "processingelement.hpp"
#include "cutoff.hpp"
#include <vector>



namespace nanos {
 
// This class initializes/finalizes the library
// All global variables MUST be declared inside
class System {
// constants
public:
   typedef enum { DEDICATED, SHARED } ExecutionMode;
   
private:
   // configuration variables
   int  numPEs;
   bool binding;
   bool profile;
   bool instrument;
   bool verboseMode;
   ExecutionMode executionMode;
   int thsPerPE;

  //cutoff policy
  cutoff * cutOffPolicy;

   std::vector<PE *> pes;

  // disable copy constructor & assignment operation
  System(const System &sys);
  const System & operator= (const System &sys);

  void config ();
  void init ();
  void start ();
  
public:
  // constructor
  System ();
  ~System ();

  void submit (WD &work);

  // methods to access configuration variables
  void setNumPEs (int npes) { numPEs = npes; }
  int getNumPEs () const { return numPEs; }
  
  void setBinding (bool set) { binding = set; }
  bool getBinding () const { return binding; }
 
  ExecutionMode getExecutionMode () const { return executionMode; }
  bool getVerbose () const { return verboseMode; }

  void setThsPerPE(int ths) { thsPerPE = ths; }
  int getThsPerPE() const { return thsPerPE; }

  bool throttleTask();
};

extern System sys;

};

#endif

