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

#ifndef _NANOS_SMP_THREAD
#define _NANOS_SMP_THREAD

#include "smpdd.hpp"
#include "basethread.hpp"
#include <pthread.h>

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{
   class SMPMultiThread;

   class SMPThread : public BaseThread
   {

         friend class SMPProcessor;

      private:
         pthread_t   _pth;
         size_t      _stackSize;
         bool        _useUserThreads;

         pthread_cond_t          _condWait;  /*! \brief Condition variable to use in pthread_cond_wait */
         static pthread_mutex_t  _mutexWait; /*! \brief Mutex to protect the sleep flag with the wait mechanism */
        
         pthread_cond_t          _completionWait;         //! Condition variable to wait for completion
         pthread_mutex_t         _completionMutex;        //! Mutex to access the completion 

         // disable copy constructor and assignment operator
         SMPThread( const SMPThread &th );
         const SMPThread & operator= ( const SMPThread &th );
        
      public:
         // constructor
         SMPThread( WD &w, PE *pe, SMPMultiThread *parent=NULL ) : BaseThread( w,pe, parent ),_stackSize(0), _useUserThreads(true) {}

         // named parameter idiom
         SMPThread & stackSize( size_t size ) { _stackSize = size; return *this; }
         SMPThread & useUserThreads ( bool use ) { _useUserThreads = use; return *this; }

         // destructor
         virtual ~SMPThread() { }

         void setUseUserThreads( bool value=true ) { _useUserThreads = value; }

         virtual void start();
         virtual void finish();
         virtual void join();
         virtual void initializeDependent( void ) {}
         virtual void runDependent ( void );

         virtual bool inlineWorkDependent( WD &work );
         virtual void preOutlineWorkDependent( WD &work ) {fatal( "SMPThread does not support preOutlineWorkDependent()" ); } ;
         virtual void outlineWorkDependent( WD &work ) {fatal( "SMPThread does not support outlineWorkDependent()" ); } ;
         virtual void switchTo( WD *work, SchedulerHelper *helper );
         virtual void exitTo( WD *work, SchedulerHelper *helper );

         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {};

         virtual void bind( void );
         

         /** \brief SMP specific yield implementation
         */
         virtual void yield();

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
          * \brief Blocks the thread if it still has enabled the sleep flag
          */
         virtual void wait();

         /*!
          * \brief Unset the flag
          */
         virtual void wakeup();

         /*!
          * \brief Waits on a condition.
          */
         virtual void block();
         
         /*! \brief Signals the thread to stop waiting. */
         virtual void unblock();
   };

   class SMPMultiThread : public SMPThread
   {
      friend class SMPProcessor;

      private:
         std::vector< BaseThread * > _threads;
         unsigned int _current;
         unsigned int _totalThreads;

         // disable copy constructor and assignment operator
         SMPMultiThread( const SMPThread &th );
         const SMPMultiThread & operator= ( const SMPMultiThread &th );

      public:
         // constructor
         SMPMultiThread( WD &w, PE *pe, unsigned int representingPEsCount, PE **representingPEs ) : SMPThread ( w, pe ), _current( 0 ), _totalThreads( representingPEsCount ) {
            setCurrentWD( w );
            _threads.reserve( representingPEsCount );
            for ( unsigned int i = 0; i < representingPEsCount; i++ )
            {
               _threads[ i ] = &( representingPEs[ i ]->startWorker( this ) );
            }
         }

         // destructor
         virtual ~SMPMultiThread() { }

         std::vector< BaseThread * >& getThreadVector() { return _threads; }

         virtual BaseThread * getNextThread()
         {
            if ( _totalThreads == 0 )
               return this;
            _current = ( _current == ( _totalThreads - 1 ) ) ? 0 : _current + 1;
            return _threads[ _current ];
         }

         unsigned int getNumThreads() const
         {
            return _totalThreads;
         }
   };
}
}

void * smp_bootthread ( void *arg );

#endif
