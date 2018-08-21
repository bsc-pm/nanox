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

#ifndef _NANOS_SMP_THREAD
#define _NANOS_SMP_THREAD

#include "smpdd.hpp"
#include "basethread_decl.hpp"
#include "processingelement_decl.hpp"
#include "system_decl.hpp"
#include <nanos-int.h>
#include "smpprocessor_fwd.hpp"
#include "pthread_decl.hpp"

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext {
   class SMPMultiThread;

   class SMPThread : public BaseThread
   {
      private:
         bool           _useUserThreads;
         PThread        _pthread;

         // disable copy constructor and assignment operator
         SMPThread( const SMPThread &th );
         const SMPThread & operator= ( const SMPThread &th );

      public:
         // constructor
         SMPThread( WD &w, PE *pe, SMPProcessor *core ) :
               BaseThread( sys.getSMPPlugin()->getNewSMPThreadId(), w, pe, NULL ), _useUserThreads( true ), _pthread(core) {}

         // named parameter idiom
         SMPThread & stackSize( size_t size );
         SMPThread & useUserThreads ( bool use ) { _useUserThreads = use; return *this; }

         // destructor
         virtual ~SMPThread() { }

         void setUseUserThreads( bool value=true ) { _useUserThreads = value; }

         virtual void initializeDependent( void ) {}
         virtual void runDependent ( void );

         virtual bool inlineWorkDependent( WD &work );
         virtual void preOutlineWorkDependent( WD &work ) { fatal( "SMPThread does not support preOutlineWorkDependent()" ); }
         virtual void outlineWorkDependent( WD &work ) { fatal( "SMPThread does not support outlineWorkDependent()" ); }
         virtual void switchTo( WD *work, SchedulerHelper *helper );
         virtual void exitTo( WD *work, SchedulerHelper *helper );

         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {};

         virtual void idle( bool debug = false );

         virtual void switchToNextThread() {
            fatal( "SMPThread does not support switchToNextThread()" );
         }

         virtual BaseThread *getNextThread()
         {
            return this;
         }

         virtual bool isCluster() { return false; }

         //virtual int checkStateDependent( int numPe ) {
         //   fatal( "SMPThread does not support checkStateDependent()" );
         //}

         /*!
          * \brief Set the flag
          */
         // PThread functions
         virtual void lock() { _pthread.mutexLock(); }
         virtual void unlock() { _pthread.mutexUnlock(); }
         virtual void initMain() { _pthread.initMain(); };
         virtual void start() { _pthread.start( this ); }
         virtual void finish() { _pthread.finish(); BaseThread::finish(); }
         virtual void join() { _pthread.join(); joined(); }
         virtual void bind() { _pthread.bind(); }
         /** \brief SMP specific yield implementation */
         virtual void yield() { _pthread.yield(); }
         /** \brief Blocks the thread if it still has enabled the sleep flag */
         virtual void wait();
         /** \brief Unset the flag */
         virtual void wakeup();
         virtual bool canBlock() { return true;}

         virtual int getCpuId() const;
#ifdef NANOS_RESILIENCY_ENABLED
         virtual void setupSignalHandlers() { _pthread.setupSignalHandlers(); }
#endif
   };

   class SMPMultiThread : public SMPThread
   {
      private:
         std::vector< BaseThread * > _threads;
         unsigned int _current;

         // disable copy constructor and assignment operator
         SMPMultiThread( const SMPThread &th );
         const SMPMultiThread & operator= ( const SMPMultiThread &th );

      public:
         // constructor
         SMPMultiThread( WD &w, SMPProcessor *pe, unsigned int representingPEsCount, PE **representingPEs );
         // destructor
         virtual ~SMPMultiThread() { }

         std::vector< BaseThread * >& getThreadVector() { return _threads; }

         virtual BaseThread * getNextThread()
         {
            if ( _threads.size() == 0 )
               return this;
            _current = ( _current == ( _threads.size() - 1 ) ) ? 0 : _current + 1;
            return _threads[ _current ];
         }

         unsigned int getNumThreads() const
         {
            return _threads.size();
         }

         void addThreadsFromPEs(unsigned int representingPEsCount, PE **representingPEs);
         virtual bool canBlock() { return false;}
         virtual void initializeDependent( void );
   };
} // namespace ext
} // namespace nanos

#endif
