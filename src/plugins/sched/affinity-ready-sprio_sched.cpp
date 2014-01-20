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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "memtracker.hpp"
#include "cache.hpp"

namespace nanos {
   namespace ext {

   bool _stealing = true;
   unsigned int _numQueues = 1;
   bool _order = true;

   // Depth propagation inside task graph
   // -1 means no depth limit
   //  0 means no propagation
   int _priorityPropagation = -1;


      class CachePrioritySchedPolicy : public SchedulePolicy
      {
         private:

            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue<>*           _readyQueues;
               WDPriorityQueue<>            _globalQueue;

               TeamData ( unsigned int size ) : ScheduleTeamData(), _globalQueue( false )
               {
                  _readyQueues = NEW WDPriorityQueue<>[size];
               }

               ~TeamData () { delete[] _readyQueues; }
            };

            /** \brief Cache Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               unsigned int _cacheId;
               ProcessingElement * _pe;
               bool _init;

               ThreadData () : _cacheId(0), _pe( NULL ), _init(false) {}
               virtual ~ThreadData () {
               }
            };

            ThreadData ** _memSpaces;

            /* disable copy and assignment */
            explicit CachePrioritySchedPolicy ( const CachePrioritySchedPolicy & );
            const CachePrioritySchedPolicy & operator= ( const CachePrioritySchedPolicy & );

         public:
            // constructor
            CachePrioritySchedPolicy() : SchedulePolicy ( "Cache-ready-priority" ) {}

            // destructor
            virtual ~CachePrioritySchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               /* Queue 0 will be the global one */
               _numQueues = sys.getCacheMap().getSize() + 1;

               _memSpaces = NEW ThreadData *[_numQueues];

               for ( unsigned int i = 0; i < _numQueues; i++ ) {
                  _memSpaces[i] = NULL;
               }

               return NEW TeamData( _numQueues );
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
            }

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               if ( !data._init ) {
                  data._cacheId = thread->runningOn()->getMemorySpaceId();
                  data._pe = thread->runningOn();
                  data._init = true;
               }
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

               if ( wd.isTied() ) {
                   unsigned int index = wd.isTiedTo()->runningOn()->getMemorySpaceId();
                   tdata._readyQueues[index].push_back ( &wd );
                   return;
               }

               // Check if there is only one memory space where this WD can be run
               unsigned int executors = 0;
               int candidate = -1;

               if ( _memSpaces[0] == NULL ) {
                  for ( int w = 0; w < sys.getNumWorkers(); w++ ) {
                     BaseThread * worker = sys.getWorker( w );
                     ThreadData * wdata = ( ThreadData * ) worker->getTeamData()->getScheduleData();

                     if ( !wdata->_init ) {
                        wdata->_cacheId = worker->runningOn()->getMemorySpaceId();
                        wdata->_pe = worker->runningOn();
                        wdata->_init = true;
                     }

                     _memSpaces[wdata->_cacheId] = wdata;
                  }
               }

               for ( unsigned int i = 0; i < _numQueues; i++ ) {
                  if ( wd.canRunIn( *_memSpaces[i]->_pe ) ) {
                     executors++;
                     candidate = _memSpaces[i]->_cacheId;
                  }
               }

               // If we found only one memory space, push this WD to its queue
               if ( executors == 1 ) {
                  tdata._readyQueues[candidate].push_back( &wd );
                  return;
               }


               tdata._globalQueue.push_back ( &wd );
            }

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void affinity_queue ( BaseThread *thread, WD &wd )
            {
                ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
                if ( !data._init ) {
                   data._cacheId = thread->runningOn()->getMemorySpaceId();
                   data._pe = thread->runningOn();
                   data._init = true;
                }
                TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

                if ( wd.isTied() ) {
                    unsigned int index = wd.isTiedTo()->runningOn()->getMemorySpaceId();
                    tdata._readyQueues[index].push_back ( &wd );
                    return;
                }

                if ( wd.getNumCopies() > 0 ) {
                   unsigned int numCaches = _numQueues - 1; //sys.getCacheMap().getSize();
                   int ranks[numCaches];
                   for (unsigned int i = 0; i < numCaches; i++ ) {
                      ranks[i] = 0;
                   }
                   CopyData * copies = wd.getCopies();
                   for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                      // Since getting the directory entry is slow, consider only outputs
                      if ( !copies[i].isPrivate() && copies[i].isOutput() ) {
                         WorkDescriptor* parent = wd.getParent();
                         if ( parent != NULL ) {
                            Directory *dir = parent->getDirectory();
                            if ( dir != NULL ) {
                               DirectoryEntry *de = dir->findEntry(copies[i].getAddress());
                               if ( de != NULL ) {
                                  Cache * cache = de->getOwner();
                                  // Give extra points if the memory space is the data's owner
                                  if ( cache != NULL && copies[i].isOutput() ) {
                                     ranks[de->getOwner()->getId()] += copies[i].getSize();
                                  }

                                  for ( unsigned int j = 0; j < numCaches; j++ ) {
                                     if ( de->getAccess( j+1 ) > 0 ) {
                                        ranks[j] += copies[i].isInput() ? copies[i].getSize() : 0;
                                        ranks[j] += copies[i].isOutput() ? copies[i].getSize() : 0;
                                     }
                                  }
                               }
                            }
                         }
                      }
                   }

                   // Do not consider those memory spaces where the WD cannot be run
                   // due to its device type
                   for (unsigned int i = 0; i < numCaches; i++ ) {
                      if ( !wd.canRunIn( *_memSpaces[i]->_pe ) ) {
                         ranks[i] = -1;
                      }
                   }

                   unsigned int winner;
                   int maxRank = 0;

                   // Alternate the visiting order (from first to last and from last to first)
                   if ( _order ) {
                      _order = false;
                      winner = 0;
                      for ( unsigned int i = 0; i < numCaches; i++ ) {
                         if ( ranks[i] > maxRank ) {
                            winner = i+1;
                            maxRank = ranks[i];
                         }
                      }
                   } else {
                      _order = true;
                      winner = numCaches - 1;
                      for ( unsigned int i = numCaches; i > 0; i-- ) {
                         if ( ranks[i-1] > maxRank ) {
                            winner = i;
                            maxRank = ranks[i-1];
                         }
                      }
                   }
                   tdata._readyQueues[winner].push_back( &wd );
                } else {
                   tdata._readyQueues[0].push_back ( &wd );
                   //fatal( "Cannot call affinity_queue() with a 0-copy WD" );
                }
            }

            virtual void atCreate ( DependableObject &depObj )
            {
               propagatePriority( &depObj, _priorityPropagation );
            }

            /*!
            *  \brief Function called when a new task must be created: the new created task
            *          is directly queued (Breadth-First policy)
            *  \param thread pointer to the thread to which belongs the new task
            *  \param wd a reference to the work descriptor of the new task
            *  \sa WD and BaseThread
            */
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               queue(thread,newWD);

               return 0;
            }

            virtual WD *atIdle ( BaseThread *thread );
            virtual WD *atBlock ( BaseThread *thread, WD *current );

            virtual WD *atAfterExit ( BaseThread *thread, WD *current )
            {
               return atBlock(thread, current );
            }

            WD * atPrefetch ( BaseThread *thread, WD &current )
            {
               // Getting the immediate successor notably increases performance
               WD * next = current.getImmediateSuccessor(*thread);

               if ( next != NULL ) return next;

               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               if ( !data._init ) {
                  data._cacheId = thread->runningOn()->getMemorySpaceId();
                  data._pe = thread->runningOn();
                  data._init = true;
               }
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

               // Try to schedule the thread with a task from its queue
               next = tdata._readyQueues[data._cacheId].pop_front ( thread );

               if ( next != NULL ) return next;

               return atIdle( thread );
            }

            WD * atBeforeExit ( BaseThread *thread, WD &current )
            {
               return atPrefetch( thread, current );//current.getImmediateSuccessor(*thread);
            }

            /*!
             * \brief This method performs the main task of the smart priority
             * scheduler, which is to propagate the priority of a WD to its
             * immediate predecessors. It is meant to be invoked from
             * DependenciesDomain::submitWithDependenciesInternal.
             * \param [in/out] predecessor The preceding DependableObject.
             * \param [in] successor DependableObject whose WD priority has to be
             * propagated.
             */
            void successorFound( DependableObject *predecessor, DependableObject *successor )
            {
               if ( predecessor == NULL ) {
                  debug( "AffinitySmartPriority::successorFound predecessor is NULL" );
                  return;
               }
               if ( successor == NULL ) {
                  debug( "AffinitySmartPriority::successorFound successor is NULL" );
                  return;
               }
               
               WD *pred = ( WD* ) predecessor->getRelatedObject();
               if ( pred == NULL ) {
                  debug( "AffinitySmartPriority::successorFound predecessor->getRelatedObject() is NULL" )
                  return;
               }
               
               WD *succ = ( WD* ) successor->getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "AffinitySmartPriority::successorFound  successor->getRelatedObject() is NULL" );
               }
               
               debug ( "Propagating priority from "
                  << (void*)succ << ":" << succ->getId() << " to "
                  << (void*)pred << ":"<< pred->getId()
                  << ", old priority: " << pred->getPriority()
                  << ", new priority: " << std::max( pred->getPriority(),
                  succ->getPriority() )
               );
               
               // Propagate priority
               if ( pred->getPriority() < succ->getPriority() ) {
                  pred->setPriority( succ->getPriority() );
                  
                  // Reorder
                  TeamData &tdata = ( TeamData & ) *nanos::myThread->getTeam()->getScheduleData();
                  // Do it for all the queues since I don't know which ones have the predecessor
                  // TODO (#652): Find a way to avoid the situation described above.
                  for ( unsigned int i = 0; i < _numQueues; i++ )
                  {
                     tdata._readyQueues[i].reorderWD( pred );
                  }
               }

#if 0
               // Propagate priority recursively
               pred->getNumDepsPredecessors();
               pred->getDOSubmit();
               predecessor->numPredecessors();
               predecessor->getSuccessors();
#endif

               propagatePriority( predecessor, _priorityPropagation - 1 );
            }

            void propagatePriority ( DependableObject * successor, int maxDepth )
            {
               if ( maxDepth == 0 ) return;

               if ( successor == NULL ) return;

               WD * succ = ( WD * ) successor->getRelatedObject();
               if ( succ == NULL ) return;

               if ( successor->numPredecessors() == 0 ) return;

               DependableObject::DependableObjectVector & predecessors = successor->getPredecessors();
               for ( DependableObject::DependableObjectVector::iterator it = predecessors.begin();
                     it != predecessors.end(); it++ ) {
                  DependableObject * obj = *it;
                  WD * pred = ( WD * ) obj->getRelatedObject();
                  if ( pred == NULL ) continue;

                  if ( pred->getPriority() < succ->getPriority() ) {

                     //std::ostringstream str;
                     //str << "Propagating priority from " << succ->getId() << " to " << pred->getId()
                     //   << ", old priority: " << pred->getPriority()
                     //   << ", new priority: " << succ->getPriority() << std::endl;
                     //std::cout << str.str();

                     pred->setPriority( succ->getPriority() );

                     // Reorder
                     TeamData &tdata = ( TeamData & ) *nanos::myThread->getTeam()->getScheduleData();
                     // Do it for all the queues since I don't know which ones have the predecessor
                     // TODO (#652): Find a way to avoid the situation described above.
                     for ( unsigned int i = 0; i < _numQueues; i++ )
                     {
                        tdata._readyQueues[i].reorderWD( pred );
                     }

                     propagatePriority( obj, maxDepth - 1 );
                  }

                  // Propagate priority
                  //std::ostringstream str2;
                  //str2 << "Calling propagate for " << pred->getId() << std::endl;
                  //std::cout << str2.str();

                  //propagatePriority( obj );
               }
            }



      };

      WD *CachePrioritySchedPolicy::atBlock ( BaseThread *thread, WD *current )
      {
         return atIdle( thread );
      }

      /*!
       */
      WD * CachePrioritySchedPolicy::atIdle ( BaseThread *thread )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         if ( !data._init ) {
            data._cacheId = thread->runningOn()->getMemorySpaceId();
            data._pe = thread->runningOn();
            data._init = true;
         }
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         /*
          *  First try to schedule the thread with a task from its queue
          */
         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
            return wd;
         } else {
            /*
             * Then try to get it from the global queue and assign it properly
             */
             wd = tdata._globalQueue.pop_front ( thread );

             if ( wd != NULL ) affinity_queue( thread, *wd );
         }

         if ( ( wd = tdata._readyQueues[data._cacheId].pop_front ( thread ) ) != NULL ) {
            return wd;
         } else if ( _stealing ) {

            wd = tdata._readyQueues[0].pop_front ( thread );

            if ( wd != NULL ) {
               /*
                * Try to get it from the general queue
                */
               return wd;
            }

            for ( unsigned int i = data._cacheId; i < sys.getCacheMap().getSize(); i++ ) {
               if ( tdata._readyQueues[i+1].size() > 1 ) {
                  wd = tdata._readyQueues[i+1].pop_back( thread );
                  return wd;
               }
            }
            for ( unsigned int i = 0; i < data._cacheId; i++ ) {
               if ( tdata._readyQueues[i+1].size() > 1 ) {
                  wd = tdata._readyQueues[i+1].pop_back( thread );
                  return wd;
               }
            }
         }

         return wd;
      }

      class CachePrioritySchedPlugin : public Plugin
      {
         public:
            CachePrioritySchedPlugin() : Plugin( "CachePriority-guided scheduling Plugin",1 ) {}

            virtual void config( Config& cfg ) {}

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW CachePrioritySchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity-ready-sp",nanos::ext::CachePrioritySchedPlugin);
