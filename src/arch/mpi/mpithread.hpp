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

#ifndef _NANOS_MPI_THREAD
#define _NANOS_MPI_THREAD

#include "mpidd.hpp"
#include "basethread.hpp"
#include "smpthread.hpp"
#include "mpiprocessor_fwd.hpp"
#include "mpispawn_fwd.hpp"

#include "request.hpp"

//TODO: Make mpi independent from pthreads? move it to OS?

namespace nanos {
namespace ext {

   class MPIThread : public SMPThread {
         friend class MPIProcessor;

      private:
         int                        _currentPE;
         mpi::RemoteSpawn*          _spawnGroup;

         // disable copy constructor and assignment operator
         MPIThread( const MPIThread &th );
         const MPIThread & operator= ( const MPIThread &th );

      public:
         // constructor
         MPIThread( WD &w, PE *pe, SMPProcessor *core) :
             SMPThread( w,pe ,core),
             _currentPE(0),
             _spawnGroup(NULL)
         {
             setLeaveTeam(true);
         }

         // destructor
         virtual ~MPIThread()
         {
         }

         void join() {
            if( !hasJoined() ) {
               SMPThread::join();
            }
         }

         virtual void runDependent ( void );

         void initializeDependent( void );

         void idle( bool debug = false );

         void addRunningPEs( MPIProcessor** pe, int nPes);

         bool switchToNextFreePE( int uuid );

         bool switchToPE(int rank, int uuid);

         virtual bool inlineWorkDependent( WD &work );

         virtual bool canBlock() { return false;}

         mpi::RemoteSpawn& getSpawnGroup() { return *_spawnGroup; }

         void setSpawnGroup( mpi::RemoteSpawn& spawn ) { _spawnGroup = &spawn; }

         void finish();

   };

} // namespace ext
} // namespace nanos

#endif // _NANOS_MPI_THREAD

