
#ifndef _NANOS_OpenCL_THREAD_DECL
#define _NANOS_OpenCL_THREAD_DECL

#include "smpthread.hpp"
#include "opencldd.hpp"

namespace nanos {
namespace ext {
    
class OpenCLThread : public SMPThread
{
private:
   OpenCLThread( const OpenCLThread &thr ); // Do not implement.
   const OpenCLThread &operator=( const OpenCLThread &thr ); // Do not implement.
   
   WD * getNextTask ( WD &wd );
   void prefetchNextTask( WD * next );

public:
   OpenCLThread( WD &wd, PE *pe ) : SMPThread( wd, pe ) {}
   ~OpenCLThread() {}
   
   void initializeDependent();
   void runDependent();
   bool inlineWorkDependent( WD &wd );
   void yield();
   void idle();

private:

   //bool checkForAbort( OpenCLDD::event_iterator i, OpenCLDD::event_iterator e );

};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_THREAD_DECL
