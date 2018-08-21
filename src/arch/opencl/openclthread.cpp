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

#include "openclprocessor.hpp"
#include "basethread.hpp"
#include "openclthread.hpp"
#include "pthread.hpp"
#include "openclevent.hpp"
#include "os.hpp"

using namespace nanos;
using namespace nanos::ext;

//
// OpenCLLocalThread implementation.
//

void OpenCLThread::initializeDependent() {
    // Since we create an OpenCLLocalThread for each OpenCLProcessor, and an
    // OpenCLProcessor for each OpenCL device, force device initialization here, in
    // order to be executed in parallel.
    OpenCLProcessor *myProc = static_cast<OpenCLProcessor *> (myThread->runningOn());
    myProc->initialize();
    setMaxPrefetch( OpenCLConfig::getPrefetchNum() );
}

void OpenCLThread::runDependent() {    
   WD &wd = getThreadWD();
   setCurrentWD( wd );
   OpenCLDD &dd = static_cast<OpenCLDD &> (wd.activateDevice(OpenCLDev));

   while ( getTeam() == NULL ) { OS::nanosleep( 100 ); }
    
   dd.getWorkFct()(wd.getData());    
   ( ( OpenCLProcessor * ) myThread->runningOn() )->cleanUp();
}

bool OpenCLThread::runWDDependent( WD &wd, GenericEvent * evt ) {
   _currKernelEvent=evt;

   OpenCLDD &dd = ( OpenCLDD & )wd.getActiveDevice();
      
   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateEvent ( NANOS_RUNTIME ) );
   NANOS_INSTRUMENT ( } else { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
   NANOS_INSTRUMENT ( } );
   ( dd.getWorkFct() )( wd.getData() );
   _currKernelEvent=NULL;
   
   NANOS_INSTRUMENT ( raiseWDClosingEvents() );

   NANOS_INSTRUMENT ( if ( wd.isRuntimeTask() ) { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( } else { );
   NANOS_INSTRUMENT (    sys.getInstrumentation()->raiseCloseStateAndBurst ( key, val ) );
   NANOS_INSTRUMENT ( } );
   return false;
}

int OpenCLThread::getCpuId() const
{
   return _pthread.getCpuId();
}

void OpenCLThread::enableWDClosingEvents ()
{
   _wdClosingEvents = true;
}

void OpenCLThread::raiseWDClosingEvents ()
{
   if ( _wdClosingEvents ) {
      NANOS_INSTRUMENT(
            Instrumentation::Event e[1];
            sys.getInstrumentation()->closeBurstEvent( &e[0],
                  sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "user-funct-location" ), 0 );

            sys.getInstrumentation()->addEventList( 1, e );
      );
      _wdClosingEvents = false;
   }
}

GenericEvent * OpenCLThread::createPreRunEvent( WD * wd )
{
   OpenCLProcessor * pe = ( OpenCLProcessor * ) this->AsyncThread::runningOn();
#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW OpenCLEvent( wd, pe->getContext(), "Pre-run event" );
#else
   return NEW OpenCLEvent( wd, pe->getContext() );
#endif
}

GenericEvent * OpenCLThread::createRunEvent( WD * wd )
{
   //unsigned int streamIdx = ( wd->getCudaStreamIdx() != -1 ) ? wd->getCudaStreamIdx() : _kernelStreamIdx;
   OpenCLProcessor * pe = ( OpenCLProcessor * ) this->AsyncThread::runningOn();

#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW OpenCLEvent( wd, pe->getContext(), "Run event" );
#else
   return NEW OpenCLEvent( wd, pe->getContext() );
#endif
}

GenericEvent * OpenCLThread::createPostRunEvent( WD * wd )
{
   OpenCLProcessor * pe = ( OpenCLProcessor * ) this->AsyncThread::runningOn();
#ifdef NANOS_GENERICEVENT_DEBUG
   return NEW OpenCLEvent( wd, pe->getContext(), "Post-run event" );
#else
   return NEW OpenCLEvent( wd, pe->getContext() );
#endif
}

void OpenCLThread::switchTo( WD *work, SchedulerHelper *helper )
{
   fatal("A Device Thread cannot call switchTo function.");
}
void OpenCLThread::exitTo( WD *work, SchedulerHelper *helper )
{
   fatal("A Device Thread cannot call exitTo function.");
}

void OpenCLThread::switchHelperDependent( WD* oldWD, WD* newWD, void *arg )
{
   fatal("A Device Thread cannot call switchHelperDependent function.");
}


void OpenCLThread::join()
{
   _pthread.join();
   joined();
}

void OpenCLThread::wait()
{
   fatal("An OpenCLThread cannot call wait function.");
}

void OpenCLThread::wakeup()
{
   // For convenience we may call wakeup for all threads, just ignore then
}

void OpenCLThread::idle( bool debug )
{
   AsyncThread::idle();
}

bool OpenCLThread::processDependentWD ( WD * wd )
{
   DOSubmit * doSubmit = wd->getDOSubmit();
   OpenCLDD& ddCurr=static_cast<OpenCLDD&>(wd->getActiveDevice());

   if ( doSubmit != NULL ) {
      DependableObject::DependableObjectVector & preds = wd->getDOSubmit()->getPredecessors();
      for ( DependableObject::DependableObjectVector::iterator it = preds.begin(); it != preds.end(); it++ ) {
         WD * wdPred = ( WD * ) it->second->getRelatedObject();         
         OpenCLDD& ddPred=static_cast<OpenCLDD&>(wdPred->getActiveDevice());
         if ( wdPred != NULL ) {
            if ( wdPred->isTiedTo() == NULL || wdPred->isTiedTo() == ( BaseThread * ) this ) {
               if ( ddPred.getOpenCLStreamIdx() != -1 ) {
                  ddCurr.setOpenclStreamIdx( ddPred.getOpenCLStreamIdx() );
                  verbose( "Setting stream for WD " << wd->getId() << " index " << ddPred.getOpenCLStreamIdx()
                        << " (from WD " << wdPred->getId() << ")" );
                  return false;
               }
            }
         }
      }
   }
   return AsyncThread::processDependentWD( wd );
}


void OpenCLThread::addEvent( GenericEvent * evt ){
   if (myThread==this) {
      AsyncThread::addEvent(evt);          
   } else {
      _evlLock.acquire();
      _externalEventsList.push_back(evt);
      _evlLock.release();
   }
}


void OpenCLThread::checkEvents(){
    if (!_externalEventsList.empty()) {
       _evlLock.acquire();
       for ( nanos::AsyncThread::GenericEventList::iterator it=_externalEventsList.begin(); 
              it != _externalEventsList.end(); ++it) {
         AsyncThread::addEvent(*it);     
       }
       _externalEventsList.clear();
       _evlLock.release();    
    }
    AsyncThread::checkEvents();         
}
