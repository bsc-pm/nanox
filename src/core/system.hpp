#ifndef _NANOS_SYSTEM
#define _NANOS_SYSTEM

#include "processingelement.hpp"
#include "cutoff.hpp"
#include <vector>
#include <string>
#include "schedule.hpp"
#include "threadteam.hpp"


namespace nanos
{

// This class initializes/finalizes the library
// All global variables MUST be declared inside

   class System
   {

         friend class Scheduler;
// constants

      public:
         typedef enum { DEDICATED, SHARED } ExecutionMode;

      private:
         // configuration variables
         int  numPEs;
         int  deviceStackSize;
         bool binding;
         bool profile;
         bool instrument;
         bool verboseMode;
         ExecutionMode executionMode;
         int thsPerPE;

         //cutoff policy and related variables
         cutoff * cutOffPolicy;
         Atomic<int> taskNum;
         Atomic<int> numReady;
         Atomic<int> idleThreads;
         Atomic<int> numTasksRunning;

         /*! names of the scheduling, cutoff and barrier plugins */
         std::string defSchedule;
         std::string defCutoff;
         std::string defBarr;

         /*! factories for scheduling, pes and barriers objects */
         sgFactory defSGFactory;
         peFactory hostFactory;
         barrFactory defBarrFactory;

         std::vector<PE *> pes;
         std::vector<BaseThread *> workers;

         // disable copy constructor & assignment operation
         System( const System &sys );
         const System & operator= ( const System &sys );

         void config ();
         void loadModules();
         void start ();
         PE * createPE ( std::string pe_type, int pid );

      public:
         // constructor
         System ();
         ~System ();

         void submit ( WD &work );
         void inlineWork ( WD &work );

         // methods to access configuration variables
         void setNumPEs ( int npes ) { numPEs = npes; }

         int getNumPEs () const { return numPEs; }

         void setDeviceStackSize ( int stackSize ) { deviceStackSize = stackSize; }

         int getDeviceStackSize () const {return deviceStackSize; }

         void setBinding ( bool set ) { binding = set; }

         bool getBinding () const { return binding; }

         ExecutionMode getExecutionMode () const { return executionMode; }

         bool getVerbose () const { return verboseMode; }

         void setThsPerPE( int ths ) { thsPerPE = ths; }

         int getThsPerPE() const { return thsPerPE; }

         int getTaskNum() const { return taskNum; }

         int getIdleNum() const { return idleThreads; }

         int getReadyNum() const { return numReady; }

         int getRunningTasks() const { return numTasksRunning; }

         // team related methods
         BaseThread * getUnassignedWorker ( void );
         ThreadTeam * createTeam ( int nthreads, SG *scheduling=NULL, void *constraints=NULL, bool reuseCurrent=true );
         void releaseWorker ( BaseThread * thread );

         //BUG: does not work: sigsegv on myThread
         int getSGSize() const { return myThread->getSchedulingGroup()->getSize(); }

         void setCutOffPolicy( cutoff * co ) { cutOffPolicy = co; }

         bool throttleTask();

         const std::string & getDefaultSchedule() const { return defSchedule; }

         const std::string & getDefaultCutoff() const { return defCutoff; }

         const std::string & getDefaultBarrier() const { return defBarr; }

         void setDefaultSGFactory ( sgFactory factory ) { defSGFactory = factory; }

         void setHostFactory ( peFactory factory ) { hostFactory = factory; }

         void setDefaultBarrFactory ( barrFactory factory ) { defBarrFactory = factory; }

   };

   extern System sys;

};

#endif

