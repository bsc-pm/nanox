/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
namespace ext {

   class MPIThread : public SMPThread
   {

         friend class MPIProcessor;

      private:        
         pthread_cond_t          _completionWait;         //! Condition variable to wait for completion
         pthread_mutex_t         _completionMutex;        //! Mutex to access the completion 
         std::vector<MPIThread*> _threadList;
         Lock _selfLock;
         std::vector<MPIProcessor*> _runningPEs;
         //Optimization so we do not search for active comms for early-release across all the nodes
         //Its a little slower but in an hipotetical distributed exascale scenario, it should be much better
         std::list<int> _ranksWithPendingComms;
         int _currPe;
         std::vector<MPIThread*>* _groupThreadList;
         Lock* _groupLock;
         Atomic<unsigned int> _selfTotRunningWds;
         Atomic<unsigned int>* _groupTotRunningWds;
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
         MPIThread( WD &w, PE *pe, SMPProcessor *core) : SMPThread( w,pe ,core), _threadList() , _selfLock(), _runningPEs(), _ranksWithPendingComms()  {
             _currPe=0;
             _selfTotRunningWds=0;
             _groupTotRunningWds=&_selfTotRunningWds;
             _groupThreadList=&_threadList;
             _groupLock=NULL;
         }
//         MPIThread( WD &w, PE *pe , MPI_Comm communicator, int rank) : BaseThread( w,pe ),_stackSize(0), _useUserThreads(true);

         // named parameter idiom
         //MPIThread & stackSize( size_t size ) { _stackSize = size; return *this; }
         //MPIThread & useUserThreads ( bool use ) { _useUserThreads = use; return *this; }

         // destructor
         virtual ~MPIThread() { 
             finish();
         }

         //void setUseUserThreads( bool value=true ) { _useUserThreads = value; }         
         
         virtual void runDependent ( void );
         void initializeDependent( void );
         
         void idle( bool debug = false );

         void addRunningPEs( MPIProcessor** pe, int nPes);
         
         bool switchToNextFreePE( int uuid );
         
         bool switchToPE(int rank, int uuid);

         virtual bool inlineWorkDependent( WD &work );
         
         virtual bool canBlock() { return false;}

         
         /**
          * Deletes an WD if no thread is executing it
          * @param wd
          * @param markToDelete If wd couldn't be deleted, add to pending list
          * @return if thread was deleted
          */
         bool deleteWd(WD* wd, bool markToDelete);
         
         /**
          * Checks which tasks have completed "input" communication and early-releases deps
          */
         void checkCommunicationsCompletion();
         
         /**
          * Checks which tasks have finished and frees/release deps them
          */
         void checkTaskEnd();
         
         void finish();
         /**
          * Frees current exrcuting WD of given PE
          * @param finishedPE
          */
         void freeCurrExecutingWD(MPIProcessor* finishedPE);
         
         void setGroupLock(Lock* gLock);
         
         Lock* getSelfLock();
         
         Atomic<unsigned int>* getSelfCounter();
         
         void setGroupCounter(Atomic<unsigned int>* gCounter);
                  
         std::vector<MPIThread*>* getSelfThreadList();
         
         void setGroupThreadList(std::vector<MPIThread*>* threadList);
         
         std::vector<MPIThread*>* getGroupThreadList();
         
         std::vector<MPIProcessor*>& getRunningPEs();
         
   };


} // namespace ext
} // namespace nanos

#endif
