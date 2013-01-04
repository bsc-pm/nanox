
#include "oclprocessor.hpp"
#include "oclthread.hpp"

using namespace nanos;
using namespace nanos::ext;

//
// OCLLocalThread implementation.
//

void OCLLocalThread::initializeDependent() {
    // Since we create an OCLLocalThread for each OCLProcessor, and an
    // OCLProcessor for each OpenCL device, force device initialization here, in
    // order to be executed in parallel.
    OCLProcessor *myProc = static_cast<OCLProcessor *> (myThread->runningOn());
    myProc->initialize();
}

void OCLLocalThread::runDependent() {
    WD &wd = getThreadWD();
    SMPDD &dd = static_cast<SMPDD &> (wd.activateDevice(SMP));

    SMPDD::work_fct workFct = dd.getWorkFct();
    workFct(wd.getData());
}

bool OCLLocalThread::inlineWorkDependent(WD &wd) {
   // Now the WD will be inminently run
   wd.start(WD::IsNotAUserLevelThread);

   OpenCLDD &dd = ( OpenCLDD & )wd.getActiveDevice();

   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
   ( dd.getWorkFct() )( wd.getData() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key ) );
   return true;
}

void OCLLocalThread::yield() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

#ifdef CLUSTER_DEV
    proc.processPendingMessages();
#endif
}

void OCLLocalThread::idle() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

#ifdef CLUSTER_DEV
    proc.processPendingMessages();
#endif
}

//bool OCLLocalThread::checkForAbort(OCLDD::event_iterator i,
//        OCLDD::event_iterator e) {
//    bool abortNeeded = false;
//
//    for (; i != e; ++i)
//        abortNeeded = abortNeeded || **i < 0;
//
//    return abortNeeded;
//}
