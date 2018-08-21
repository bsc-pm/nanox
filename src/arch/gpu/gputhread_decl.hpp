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

#ifndef _NANOS_GPU_THREAD_DECL
#define _NANOS_GPU_THREAD_DECL

//#include "compatibility.hpp"
//#include "gpudd.hpp"
#include "genericevent_decl.hpp"
#include "asyncthread_decl.hpp"
#include "gpuprocessor_fwd.hpp"
#include "smpthread.hpp"



namespace nanos {
namespace ext {

   class GPUThread : public nanos::AsyncThread
   {
      private:
         int                           _gpuDevice; // Assigned GPU device Id
         int                           _kernelStreamIdx;
         bool                          _wdClosingEvents; //! controls whether an instrumentation event should be generated at WD completion
         void *                        _cublasHandle; //! Context pointer for CUBLAS library
         void *                        _cusparseHandle; //! Context pointer for cuSPARSE library
         BaseThread *                  _cudaThreadInst;
         PThread                       _pthread;
         unsigned int _prefetchedWDs;

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
         GPUThread( WD &w, PE *pe, SMPProcessor *core, int device ) :
            AsyncThread( sys.getSMPPlugin()->getNewSMPThreadId(), w, pe ), _gpuDevice( device ), _kernelStreamIdx ( 0 ),
               _wdClosingEvents( false ), _cublasHandle( NULL ), _cusparseHandle( NULL ), _cudaThreadInst( NULL ), _pthread( core ), _prefetchedWDs(0) { setCurrentWD( w ); }

         // destructor
         ~GPUThread() {}

         void initializeDependent( void );
         void runDependent ( void );

         void preOutlineWorkDependent( WD &work ) { fatal( "GPUThread does not support preOutlineWorkDependent()" ); }
         void outlineWorkDependent( WD &work ) { fatal( "GPUThread does not support outlineWorkDependent()" ); }

         bool runWDDependent( WD &work, GenericEvent * evt = NULL );
         //bool inlineWorkDependent( WD &work );

         bool processDependentWD ( WD * wd );

         virtual void idle( bool debug );

         void processTransfers();

         unsigned int getCurrentKernelExecStreamIdx();

         int getGPUDevice ();

         void * getCUBLASHandle();
         unsigned int getPrefetchedWDsCount() const;

         void * getCUSPARSEHandle();

         BaseThread * getCUDAThreadInst();
         void setCUDAThreadInst( BaseThread * thread );

         GenericEvent * createPreRunEvent( WD * wd );
         GenericEvent * createRunEvent( WD * wd );
         GenericEvent * createPostRunEvent( WD * wd );

         // PThread functions
         virtual void start() { _pthread.start( this ); }
         virtual void finish() { _pthread.finish(); BaseThread::finish(); }
         virtual void join();
         virtual void bind() { _pthread.bind(); }
         /** \brief GPU specific yield implementation */
         virtual void yield();
         /** \brief Blocks the thread if it still has enabled the sleep flag */
         virtual void wait();
         /** \brief Unset the flag */
         virtual void wakeup();
         virtual int getCpuId() const;
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

         void raiseKernelLaunchEvent ();
         void closeKernelLaunchEvent ();

         void raiseWDRunEvent ( WD * wd );
         void closeWDRunEvent ( WD * wd );

         void raiseAsyncInputEvent ( size_t size );
         void closeAsyncInputEvent ( size_t size );

         void raiseAsyncOutputEvent ( size_t size );
         void closeAsyncOutputEvent ( size_t size );

   };

} // namespace ext
} // namespace nanos

#endif
