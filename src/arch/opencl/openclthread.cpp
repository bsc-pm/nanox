
/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "openclprocessor.hpp"
#include "openclthread.hpp"

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
}

void OpenCLThread::runDependent() {    
    WD &wd = getThreadWD();
    setCurrentWD( wd );
    OpenCLDD &dd = static_cast<OpenCLDD &> (wd.activateDevice(OpenCLDev));

    dd.getWorkFct()(wd.getData());    
   ( ( OpenCLProcessor * ) myThread->runningOn() )->cleanUp();
}

bool OpenCLThread::inlineWorkDependent(WD &wd) {
   // Now the WD will be inminently run
   wd.start(WD::IsNotAUserLevelThread);

   OpenCLDD &dd = ( OpenCLDD & )wd.getActiveDevice();
   
   
   OpenCLProcessor *myProc = static_cast<OpenCLProcessor *> (myThread->runningOn());
   myProc->waitForEvents();

   ( dd.getWorkFct() )( wd.getData() );
   
   NANOS_INSTRUMENT ( raiseWDClosingEvents() );
   return true;
}

void OpenCLThread::yield() {
    //OpenCLProcessor &proc = *static_cast<OpenCLProcessor *> (myThread->runningOn());

   // proc.execTransfers();

}

void OpenCLThread::idle() {
    //OpenCLProcessor &proc = *static_cast<OpenCLProcessor *> (myThread->runningOn());

   // proc.execTransfers();

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
//bool OpenCLLocalThread::checkForAbort(OpenCLDD::event_iterator i,
//        OpenCLDD::event_iterator e) {
//    bool abortNeeded = false;
//
//    for (; i != e; ++i)
//        abortNeeded = abortNeeded || **i < 0;
//
//    return abortNeeded;
//}
