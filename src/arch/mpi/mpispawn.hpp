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

#ifndef NANOS_MPISPAWN_HPP
#define NANOS_MPISPAWN_HPP

#include "mpiplugin.hpp"
#include "mpiprocessor.hpp"
#include "mpithread.hpp"

#include "finish.hpp"

#include "mutex.hpp"
#include "lock.hpp"
#include "filelock.hpp"

#include "schedule.hpp"

#include <vector>

#include <mpi.h>

namespace nanos {
namespace mpi {

// Analogous to SMPMultiThread
class RemoteSpawn {
	private:
		MPI_Comm                   _communicator;
		MPI_Comm                   _intercommunicator;

		std::vector<nanos::ext::MPIProcessor*> _remotes;
		std::vector<ext::MPIThread*>    _threads;

		Atomic<unsigned>           _runningWDs;
		Lock                       _lock;

		// Not copyable nor copy-assignable
		RemoteSpawn( const RemoteSpawn& );

		RemoteSpawn& operator=( const RemoteSpawn& );

	public:
		RemoteSpawn( size_t helperThreadNumber, MPI_Comm communicator,
		             MPI_Comm intercommunicator,
		             std::vector<ext::MPIProcessor*> remotes ) :
			_communicator( communicator ),
			_intercommunicator( intercommunicator ),
			_remotes( remotes ),
			_threads(),
			_runningWDs( 0 ),
			_lock()
		{
			ensure0( helperThreadNumber <= _remotes.size(), "Too many helper threads requested" );

			_threads.reserve( helperThreadNumber );
			for( unsigned t = 0; t < helperThreadNumber; ++t ) {
				_threads.push_back( &_remotes[t]->startMPIThread(NULL) );
				_threads.back()->setSpawnGroup( *this );
			}

			// Start the threads...
			std::vector<ProcessingElement*> processors(_remotes.begin(), _remotes.end() );
			std::vector<BaseThread*> threads( _threads.begin(), _threads.end() );

			sys.addPEsAndThreadsToTeam( processors.data(), processors.size(),
			                            threads.data(), threads.size() );

			ext::MPIPlugin::addPECount( _remotes.size() );
			ext::MPIPlugin::addWorkerCount( _threads.size() );
		}

		MPI_Comm getCommunicator() { return _communicator; }

		MPI_Comm getIntercommunicator() { return _intercommunicator; }

		std::vector<ext::MPIThread*>& getThreads() { return _threads; }

		std::vector<ext::MPIProcessor*>& getRemoteProcessors() { return _remotes; }

		bool isMaster( const ext::MPIThread& thread ) const
		{
			return _threads.front() == &thread;
		}

		std::vector<mpi::request> getPendingTaskEndRequests() {
			std::vector<mpi::request> pendingTaskEnd;
			pendingTaskEnd.reserve( _remotes.size() );

			std::vector<ext::MPIProcessor*>::iterator peIterator;
			for( peIterator = _remotes.begin(); peIterator != _remotes.end(); ++peIterator ) {
				pendingTaskEnd.push_back( (*peIterator)->getTaskEndRequest() );
			}

			return pendingTaskEnd;
		}

		void registerTaskInit() {
			_runningWDs++;
		}

		std::vector<int> waitFinishedTasks() {
			UniqueLock<Lock> guard( _lock, nanos::try_to_lock );
			std::vector<int> finishedTaskEndIds;

			if( _runningWDs > 0 ) {
				std::vector<mpi::request> pendingTaskEnd;
				pendingTaskEnd = getPendingTaskEndRequests();

				finishedTaskEndIds = mpi::request::wait_some( pendingTaskEnd );

				if( !finishedTaskEndIds.empty() && guard.owns_lock() ) {

					std::vector<int>::iterator taskEndIdIter;
					for( taskEndIdIter = finishedTaskEndIds.begin();
						 taskEndIdIter != finishedTaskEndIds.end(); ++taskEndIdIter ) {

						ext::MPIProcessor* finishedPE = _remotes.at( *taskEndIdIter );
						myThread->setRunningOn(finishedPE);

						WD* finishedWD = finishedPE->freeCurrExecutingWd();
						if( finishedWD != NULL ) { // PE already released?
							_runningWDs--;

							//Finish the wd, finish work
							WD* previousWD = myThread->getCurrentWD();
							myThread->setCurrentWD( *finishedWD );

							finishedWD->releaseInputDependencies();
							finishedWD->finish();
							Scheduler::finishWork( finishedWD, true );

							myThread->setCurrentWD(*previousWD);

							// Destroy wd
							finishedWD->~WorkDescriptor();
							delete[] (char *)finishedWD;
						}
					}
				}
			}
			return finishedTaskEndIds;
		}

		void finalize() {
			waitFinishedTasks();

			//Synchronize parents before killing shared resources (as each parent only waits for his task
			//this prevents one parent killing a "son" which is still executing things from other parents)
			int rank = MPI_PROC_NULL;
			MPI_Comm_rank( _communicator, &rank );
			MPI_Barrier( _communicator );

			//De-spawn threads
			std::vector<ext::MPIThread*>::iterator itThread;
			for( itThread = _threads.begin(); itThread != _threads.end(); ++itThread ) {
				ext::MPIThread* mpiThread = *itThread;
				mpiThread->lock();
				mpiThread->stop();
				mpiThread->join();
				mpiThread->unlock();
			}

			std::vector<ext::MPIProcessor*>::iterator itRemote;
			for( itRemote = _remotes.begin(); itRemote != _remotes.end() ; ++itRemote) {
				ext::MPIProcessor* remote = *itRemote;
				//Only owner will send kill signal to the worker
				if ( remote->isOwner() ) {
					mpi::command::Finish::Requestor finishCommand( *remote );
					finishCommand.dispatch();
				}

				//If im the root do all the automatic control file work ONCE (first remote node), this time free the hosts
				if( remote->getRank()==0 && rank == 0 && !ext::MPIProcessor::getMpiControlFile().empty() ) {
					//PPH List is the list of hosts which were consumed by the spawn of these ranks, lets "free" them
					int* pph_list = remote->getPphList();
					if( pph_list != NULL ) {
						std::string controlName = ext::MPIProcessor::getMpiControlFile();
						FileMutex mutex( const_cast<char*>(controlName.c_str()) );
						mutex.lock();

						FILE* file = fdopen( mutex.native_handle(), "r+" );
						size_t num_bytes=0;
						for ( int i=0; pph_list[i] != -1 ; ++i ) {
							if (pph_list[i]!=0) {
								fseek( file, num_bytes, SEEK_SET );
								const int freeNode=0;
								fprintf( file, "%d\n", freeNode );
							}
							num_bytes+=2;
						}
						fclose( file );

						remote->setPphList( NULL );
						delete[] pph_list;
						mutex.unlock();
					}
				}
				sys.getPEs().erase( remote->getId() );
				delete remote;
			}
#ifdef OPEN_MPI
			MPI_Comm_free(&_intercommunicator);
#else
			MPI_Comm_disconnect(&_intercommunicator);
#endif
		}
};

} // namespace mpi
} // namespace nanos

#endif // NANOS_MPISPAWN_HPP
