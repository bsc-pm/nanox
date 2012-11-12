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

#ifndef _NANOS_MPI_THREAD
#define _NANOS_MPI_THREAD

#include "mpidd.hpp"
#include "basethread.hpp"
#include <pthread.h>

//TODO: Make mpi independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{

   class MPIThread : public BaseThread
   {

         friend class MPIProcessor;

      private:
         pthread_t   _pth;
         size_t      _stackSize;
         bool        _useUserThreads;
//         MPI_Comm _communicator;
//         int _rank;

         // disable copy constructor and assignment operator
         MPIThread( const MPIThread &th );
         const MPIThread & operator= ( const MPIThread &th );

      public:
         static bool _mpiThreadLaunched;
         // constructor
         MPIThread( WD &w, PE *pe ) : BaseThread( w,pe ),_stackSize(0), _useUserThreads(true) {}
//         MPIThread( WD &w, PE *pe , MPI_Comm communicator, int rank) : BaseThread( w,pe ),_stackSize(0), _useUserThreads(true);

         // named parameter idiom
         MPIThread & stackSize( size_t size ) { _stackSize = size; return *this; }
         MPIThread & useUserThreads ( bool use ) { _useUserThreads = use; return *this; }

         // destructor
         virtual ~MPIThread() { }

         void setUseUserThreads( bool value=true ) { _useUserThreads = value; }         
         void workerMpiLoop();
         
         virtual void start();
         virtual void join();
         virtual void initializeDependent( void ) {}
         virtual void runDependent ( void );

         virtual bool inlineWorkDependent( WD &work );
         virtual void switchTo( WD *work, SchedulerHelper *helper );
         virtual void exitTo( WD *work, SchedulerHelper *helper );

         virtual void switchHelperDependent( WD* oldWD, WD* newWD, void *arg );
         virtual void exitHelperDependent( WD* oldWD, WD* newWD, void *arg ) {};

         virtual void bind( void );

         /** \brief MPI specific yield implementation
         */
         virtual void yield();
   };


}
}

void * mpi_bootthread ( void *arg );

#endif
