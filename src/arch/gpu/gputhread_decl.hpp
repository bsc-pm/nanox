/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_GPU_THREAD_DECL
#define _NANOS_GPU_THREAD_DECL

//#include "compatibility.hpp"
//#include "gpudd.hpp"
#include "genericevent_decl.hpp"
#include "asyncthread_decl.hpp"
#include "gpuprocessor_fwd.hpp"
#include "smpthread.hpp"

#include <pthread.h>


namespace nanos {
namespace ext
{

   class GPUThread : public nanos::AsyncThread
   {
      private:
         pthread_t                     _pth;
         int                           _gpuDevice; // Assigned GPU device Id
         int                           _kernelStreamIdx;
         bool                          _wdClosingEvents; //! controls whether an instrumentation event should be generated at WD completion
         void *                        _cublasHandle; //! Context pointer for CUBLAS library

         // disable copy constructor and assignment operator
         GPUThread( const GPUThread &th );
         const GPUThread & operator= ( const GPUThread &th );

         void raiseWDClosingEvents ( WD * wd );

         WD * getNextTask ( WD &wd );
         void prefetchNextTask( WD * next );
         void executeRequestedTransfers( GPUProcessor & myGPU );
         void executeOutputTransfers( GPUProcessor & myGPU );

      public:
         // constructor
         GPUThread( WD &w, PE *pe, int device ) : AsyncThread( w, pe ), _gpuDevice( device ), _kernelStreamIdx ( 0 ),
         _wdClosingEvents( false ), _cublasHandle( NULL ) {}

         // destructor
         ~GPUThread() {}

         void initializeDependent( void );
         void runDependent ( void );

         bool runWDDependent( WD &work );
         //bool inlineWorkDependent( WD &work );

         void yield();

         void idle();

         void processTransfers();

         unsigned int getCurrentKernelExecStreamIdx();

         int getGPUDevice ();
         void enableWDClosingEvents ();

         void * getCUBLASHandle();

         GenericEvent * createPreRunEvent( WD * wd );
         GenericEvent * createRunEvent( WD * wd );
         GenericEvent * createPostRunEvent( WD * wd );


         void start();
         void join();

         void switchTo( WD *work, SchedulerHelper *helper );
         void exitTo( WD *work, SchedulerHelper *helper );

         void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {}

   };

   void * gpu_bootthread ( void *arg );

}
}

#endif
