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

#ifndef _NANOS_PTHREAD_DECL
#define _NANOS_PTHREAD_DECL

#include "smpprocessor_fwd.hpp"
#include "taskexecutionexception_decl.hpp"
#include <pthread.h>
#include <signal.h>


namespace nanos {

   class PThread
   {
      friend class ext::SMPProcessor;

      private:
         ext::SMPProcessor * _core;

         pthread_t   _pth;
         size_t      _stackSize;

         pthread_cond_t    _condWait;  /*! \brief Condition variable to use in pthread_cond_wait */
         pthread_mutex_t   _mutexWait; /*! \brief Mutex to protect the sleep flag with the wait mechanism */

         // disable copy constructor and assignment operator
         PThread( const PThread &th );
         const PThread & operator= ( const PThread &th );

      public:
         // constructor
         PThread( ext::SMPProcessor * core ) : _core( core ), _pth( ), _stackSize( 0 ) { }

         // destructor
         virtual ~PThread() {}

         int getCpuId() const;

         size_t getStackSize ();
         void setStackSize( size_t size );

         virtual void initMain();
         virtual void start( BaseThread * th );
         virtual void finish();
         virtual void join();

         virtual void bind();

         virtual void yield();

         virtual void mutexLock();
         virtual void mutexUnlock();

         virtual void condWait();
         virtual void condSignal();

#ifdef NANOS_RESILIENCY_ENABLED
         virtual void setupSignalHandlers();
#endif
   };

} // namespace nanos

void * os_bootthread ( void *arg );

#ifdef NANOS_RESILIENCY_ENABLED
void taskExecutionHandler(int sig, siginfo_t* si, void* context)
   throw (TaskExecutionException);
#endif

#endif
