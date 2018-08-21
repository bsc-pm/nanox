/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#ifndef _NANOS_OpenCL_THREAD_DECL
#define _NANOS_OpenCL_THREAD_DECL

#include "smpthread.hpp"
#include "opencldd.hpp"
#include "genericevent_decl.hpp"
#include "asyncthread_decl.hpp"

namespace nanos {
namespace ext {
    
class OpenCLThread : public nanos::AsyncThread
{
private:
   bool _wdClosingEvents; //! controls whether an instrumentation event should be generated at WD completion
   PThread _pthread;
   GenericEvent* _currKernelEvent;
   nanos::AsyncThread::GenericEventList _externalEventsList;
   Lock _evlLock;
   int _openclStreamIdx;
   
   OpenCLThread( const OpenCLThread &thr ); // Do not implement.
   const OpenCLThread &operator=( const OpenCLThread &thr ); // Do not implement.
   
   WD * getNextTask ( WD &wd );
   void prefetchNextTask( WD * next );
   void raiseWDClosingEvents ();
   
public:
   OpenCLThread( WD &wd, PE *pe, SMPProcessor* core ) : nanos::AsyncThread( sys.getSMPPlugin()->getNewSMPThreadId(), wd, pe ), _pthread(core), _externalEventsList(), _openclStreamIdx(1) 
   { }

   ~OpenCLThread() {}
   
   GenericEvent* getCurrKernelEvent() { return _currKernelEvent; };
   void initializeDependent();
   void runDependent();
   void enableWDClosingEvents ();
   
   void preOutlineWorkDependent( WD &work ) { fatal( "GPUThread does not support preOutlineWorkDependent()" ); }
   void outlineWorkDependent( WD &work ) { fatal( "GPUThread does not support outlineWorkDependent()" ); }
   
   bool runWDDependent( WD &wd, GenericEvent * evt );
   bool processDependentWD ( WD * wd );

   GenericEvent * createPreRunEvent( WD * wd );
   GenericEvent * createRunEvent( WD * wd );
   GenericEvent * createPostRunEvent( WD * wd );
   
   
   // PThread functions
   virtual void start() { _pthread.start( this ); }
   virtual void finish() { _pthread.finish(); BaseThread::finish(); }
   virtual void join();
   virtual void bind() { _pthread.bind(); }
   /** \brief GPU specific yield implementation */
   virtual void yield() { AsyncThread::yield(); _pthread.yield(); }
   /** \brief Blocks the thread if it still has enabled the sleep flag */
   virtual void wait();
   /** \brief Unset the flag */
   virtual void wakeup();
   virtual int getCpuId() const;   
   virtual void idle( bool debug );
   #ifdef NANOS_RESILIENCY_ENABLED
      virtual void setupSignalHandlers() { _pthread.setupSignalHandlers(); }
   #endif
   
   
   
   void switchTo( WD *work, SchedulerHelper *helper );
   void exitTo( WD *work, SchedulerHelper *helper );

   void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
   void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {}

   void switchToNextThread() { fatal( "GPUThread does not support switchToNextThread()" ); }
   BaseThread *getNextThread() { return this; }
   bool isCluster() { return false; }
   
   //TODO: CHECK if this should be the base implementation of Async Thread
   virtual void addEvent( GenericEvent * evt );
   virtual void checkEvents();

private:

   //bool checkForAbort( OpenCLDD::event_iterator i, OpenCLDD::event_iterator e );

};

} // namespace ext
} // namespace nanos

#endif // _NANOS_OpenCL_THREAD_DECL
