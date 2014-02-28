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
#include "smpthread.hpp"
#include "mpiprocessor_fwd.hpp"

//TODO: Make mpi independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{

   class MPIThread : public SMPThread
   {

         friend class MPIProcessor;

      private:
         pthread_t   _pth;
         std::vector<MPIProcessor*> _runningPEs;
         int _currPe;
         Lock _selfLock;
         Lock* _groupLock;
         Atomic<int> _totRunningWds;
         Atomic<int>* _groupTotRunningWds;
         std::vector<MPIThread*> _threadList;
         std::vector<MPIThread*>* _groupThreadList;
         std::list<WD*> _wdMarkedToDelete;
         
         //size_t      _stackSize;
         //bool        _useUserThreads;         
//         MPI_Comm _communicator;
//         int _rank;

         // disable copy constructor and assignment operator
         MPIThread( const MPIThread &th );
         const MPIThread & operator= ( const MPIThread &th );

      public:
         // constructor
         MPIThread( WD &w, PE *pe ) : _threadList() , _selfLock(), _runningPEs(), SMPThread( w,pe ) {
             _currPe=0;
             _totRunningWds=0;
             _groupTotRunningWds=&_totRunningWds;
             _groupThreadList=&_threadList;
             _groupLock=NULL;
         }
//         MPIThread( WD &w, PE *pe , MPI_Comm communicator, int rank) : BaseThread( w,pe ),_stackSize(0), _useUserThreads(true);

         // named parameter idiom
         //MPIThread & stackSize( size_t size ) { _stackSize = size; return *this; }
         //MPIThread & useUserThreads ( bool use ) { _useUserThreads = use; return *this; }

         // destructor
         virtual ~MPIThread() { }

         //void setUseUserThreads( bool value=true ) { _useUserThreads = value; }         
         
         virtual void runDependent ( void );
         void initializeDependent( void );
         
         void idle();

         void addRunningPEs( MPIProcessor** pe, int nPes);
         
         bool switchToNextFreePE( int uuid );
         
         bool switchToPE(int rank, int uuid);

         virtual bool inlineWorkDependent( WD &work );
         
         /**
          * Deletes an WD if no thread is executing it
          * @param wd
          * @param markToDelete If wd couldn't be deleted, add to pending list
          * @return if thread was deleted
          */
         bool deleteWd(WD* wd, bool markToDelete);
         
         void checkTaskEnd();
         
         void bind();
         
         void finish();
         /**
          * Frees current exrcuting WD of given PE
          * @param finishedPE
          */
         void freeCurrExecutingWD(MPIProcessor* finishedPE);
         
         void setGroupLock(Lock* gLock);
         
         Lock* getSelfLock();
         
         Atomic<int>* getSelfCounter();
         
         void setGroupCounter(Atomic<int>* gCounter);
                  
         std::vector<MPIThread*>* getSelfThreadList();
         
         void setGroupThreadList(std::vector<MPIThread*>* threadList);
         
         std::vector<MPIProcessor*>& getRunningPEs();

   };


}
}

#endif
