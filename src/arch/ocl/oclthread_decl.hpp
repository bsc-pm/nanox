
#ifndef _NANOS_OCL_THREAD_DECL
#define _NANOS_OCL_THREAD_DECL

#include "smpthread.hpp"
#include "ocldd.hpp"

namespace nanos {
namespace ext {

class OCLThread {
protected:
   OCLThread() { }

   OCLThread( const OCLThread &thr ); // Do not implement.
   const OCLThread &operator=( const OCLThread &thr ); // Do not implement.
};

class OCLLocalThread : public OCLThread, public SMPThread
{
public:
  class UserEventWaiter : public SingleSyncCond<EqualConditionChecker<bool> >
  {
  public:
     UserEventWaiter() : _signalled(false),
                         _status(-1) {
       setConditionChecker( EqualConditionChecker<bool>( &_signalled, true ) );
     }

     // Do not implement.
     UserEventWaiter(const UserEventWaiter &waiter);

     // Do not implement.
     const UserEventWaiter &operator=(const UserEventWaiter &waiter);

  public:
     virtual void signal(int status) {
       _signalled = true;
       _status = status;
       SingleSyncCond<EqualConditionChecker<bool> >::signal();
     }

     int getStatus() const { return _status; }

  private:
     volatile bool _signalled;
     volatile int _status;
  };

public:
   OCLLocalThread( WD &wd, PE *pe ) : SMPThread( wd, pe )
   {
      setCurrentWD( wd );
   }

   OCLLocalThread( const OCLThread &thr ); // Do not implement.
   const OCLLocalThread &operator=( const OCLThread &thr ); // Do not implement.

public:
   virtual void initializeDependent();
   virtual void runDependent();
   virtual void inlineWorkDependent( WD &wd );
   virtual void yield();
   virtual void idle();

private:
   void inlineWorkDependent( OCLNDRangeKernelStarSSDD &dd,
                             OCLNDRangeKernelStarSSDD::Data &data,
                             OCLNDRangeKernelStarSSDD::arg_iterator j,
                             OCLNDRangeKernelStarSSDD::arg_iterator f);

   bool checkForAbort( OCLDD::event_iterator i, OCLDD::event_iterator e );

private:
   std::map<void *, UserEventWaiter *> _pendingUserEvents;
};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_THREAD_DECL
