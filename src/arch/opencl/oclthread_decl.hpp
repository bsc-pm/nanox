
#ifndef _NANOS_OCL_THREAD_DECL
#define _NANOS_OCL_THREAD_DECL

#include "smpthread.hpp"
#include "ocldd.hpp"

namespace nanos {
namespace ext {
    
class OCLThread : public SMPThread
{
public:
   OCLThread( WD &wd, PE *pe ) : SMPThread( wd, pe ) {}

   OCLThread( const OCLThread &thr ); // Do not implement.
   const OCLThread &operator=( const OCLThread &thr ); // Do not implement.

public:
   void initializeDependent();
   void runDependent();
   bool inlineWorkDependent( WD &wd );
   void yield();
   void idle();

private:

   //bool checkForAbort( OpenCLDD::event_iterator i, OCLDD::event_iterator e );

};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_THREAD_DECL
