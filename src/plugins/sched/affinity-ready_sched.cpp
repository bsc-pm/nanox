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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "os.hpp"
#include "memtracker.hpp"
#include "regioncache.hpp"
#include "regiondirectory.hpp"

//#define EXTRA_QUEUE_DEBUG

namespace nanos {
   namespace ext {

      class ReadyCacheSchedPolicy : public SchedulePolicy
      {
         class ThreadData;

         public:
            typedef std::set<ThreadData *> ThreadDataSet;
            using SchedulePolicy::queue;

         private:

            class SchedQueues
            {
               public:
                  SchedQueues() {}

                  SchedQueues( int memSpaces ) {}
                  virtual ~SchedQueues() {}

                  virtual size_t size( int index ) = 0;

                  virtual void globalPushBack ( WD * wd ) = 0;
                  virtual WD * globalPopFront ( BaseThread * thread ) = 0;
                  virtual void pushBack ( WD * wd, int index ) = 0;
                  virtual WD * popFront ( BaseThread * thread, int index ) = 0;
                  virtual WD * popBack ( BaseThread * thread, int index ) = 0;
                  virtual WD * fetchWD ( BaseThread * thread, int memId ) = 0;
                  virtual bool reorderWD ( WD * wd, int index ) { return false; }
            };

            class SchedQueuesWDQ : public SchedQueues
            {
               WDDeque   _globalReadyQueue;
               WDDeque * _readyQueues;

#ifdef EXTRA_QUEUE_DEBUG
               PE     ** _pes;
#endif

               public:
                  SchedQueuesWDQ( int memSpaces ) : SchedQueues(), _globalReadyQueue( /* enableDeviceCounter */ false )
                  {
                     _readyQueues = NEW WDDeque[memSpaces];

#ifdef EXTRA_QUEUE_DEBUG
                     _pes = NEW PE*[memSpaces];
                     for ( int i = 0; i < memSpaces; i++) {
                        PE &pe = sys.getPEWithMemorySpaceId( i );
                        _pes[i] = &pe;
                     }
#endif
                  }

                  ~SchedQueuesWDQ()
                  {
                     delete[] _readyQueues;

#ifdef EXTRA_QUEUE_DEBUG
                     delete [] _pes;
#endif
                  }

                  inline size_t size( int index )
                  {
                     return _readyQueues[index].size();
                  }

                  inline void globalPushBack ( WD * wd )
                  {
                     _globalReadyQueue.push_back( wd );
                  }

                  inline WD * globalPopFront ( BaseThread * thread )
                  {
                     return _globalReadyQueue.pop_front( thread );
                  }

                  inline void pushBack ( WD * wd, int index )
                  {
#ifdef EXTRA_QUEUE_DEBUG
                     if ( !_pes[index]->canRun( *wd ) ) {
                        std::cout << "Impossible to add WD to incompatible queue!!!" << std::endl;
                     }
#endif

                     _readyQueues[index].push_back( wd );
                  }

                  inline WD * popFront ( BaseThread * thread, int index )
                  {
                     return _readyQueues[index].pop_front( thread );
                  }

                  inline WD * popBack ( BaseThread * thread, int index )
                  {
                     return _readyQueues[index].pop_back( thread );
                  }

                  WD * fetchWD ( BaseThread * thread, int memId ) {

                     NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)

                     WD * wd = NULL;

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< NoCopy > ( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( !_noInvalAware ) {
                        if ( ( wd = _readyQueues[memId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, Not< NoCopy > > > ( thread ) ) != NULL ) {
                           NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                           NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                           return wd;
                        }
                     }

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, NoCopy > >( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPYNOOUTMEM;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< WouldNotRunOutOfMemory >( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOOUTMEM;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( ( wd = _readyQueues[memId].pop_front( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     return wd;
                  }

            };

            struct SchedQueuesWDPQ : public SchedQueues
            {
               WDPriorityQueue<>   _globalReadyQueue;
               WDPriorityQueue<> * _readyQueues;

#ifdef EXTRA_QUEUE_DEBUG
               PE     ** _pes;
#endif

               public:
                  SchedQueuesWDPQ( int memSpaces ) : SchedQueues(), _globalReadyQueue( /* optimise option */ true )
                  {
                     _readyQueues = NEW WDPriorityQueue<>[memSpaces];

#ifdef EXTRA_QUEUE_DEBUG
                     _pes = NEW PE*[memSpaces];
                     for ( int i = 0; i < memSpaces; i++) {
                        PE &pe = sys.getPEWithMemorySpaceId( i );
                        _pes[i] = &pe;
                     }
#endif
                 }

                  ~SchedQueuesWDPQ()
                  {
                     delete[] _readyQueues;

#ifdef EXTRA_QUEUE_DEBUG
                     delete [] _pes;
#endif
                  }

                  inline size_t size( int index )
                  {
                     return _readyQueues[index].size();
                  }

                  inline void globalPushBack ( WD * wd )
                  {
                     _globalReadyQueue.push_back( wd );
                  }

                  inline WD * globalPopFront ( BaseThread * thread )
                  {
                     return _globalReadyQueue.pop_front( thread );
                  }

                  inline void pushBack ( WD * wd, int index )
                  {
#ifdef EXTRA_QUEUE_DEBUG
                     if ( !_pes[index]->canRun( *wd ) ) {
                        std::cout << "Impossible to add WD to incompatible queue!!!" << std::endl;
                     }
#endif

                     _readyQueues[index].push_back( wd );
                  }

                  inline WD * popFront ( BaseThread * thread, int index )
                  {
                     return _readyQueues[index].pop_front( thread );
                  }

                  inline WD * popBack ( BaseThread * thread, int index )
                  {
                     return _readyQueues[index].pop_back( thread );
                  }

                  WD * fetchWD ( BaseThread * thread, int memId ) {

                     NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("sched-affinity-constraint");)

                     WD * wd = NULL;

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< NoCopy > ( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPY;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( !_noInvalAware ) {
                        if ( ( wd = _readyQueues[memId].popFrontWithConstraints< And < WouldNotTriggerInvalidation, Not< NoCopy > > > ( thread ) ) != NULL ) {
                           NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPY;)
                           NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                           return wd;
                        }
                     }

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< And < WouldNotRunOutOfMemory, NoCopy > >( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCOPYNOOUTMEM;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( ( wd = _readyQueues[memId].popFrontWithConstraints< WouldNotRunOutOfMemory >( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = SICOPYNOOUTMEM;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     if ( ( wd = _readyQueues[memId].pop_front( thread ) ) != NULL ) {
                        NANOS_INSTRUMENT(static nanos_event_value_t val = NOCONSTRAINT;)
                        NANOS_INSTRUMENT(sys.getInstrumentation()->raisePointEvents( 1, &key, &val );)
                        return wd;
                     }

                     return wd;
                  }

                  inline bool reorderWD ( WD * wd, int index )
                  {
                     return _readyQueues[index].reorderWD( wd );
                  }

            };

            struct TeamData : public ScheduleTeamData
            {
               SchedQueues      * _queues;
               size_t*            _createdData;
               unsigned int       _numMemSpaces;
               unsigned int     * _load;
               ThreadDataSet      _teamThreadData;
               Lock               _teamThrDataLock;

               TeamData ( unsigned int size ) : ScheduleTeamData(), _teamThreadData(), _teamThrDataLock()
               {
                  // +1 to count the host memory space as well
                  _numMemSpaces = sys.getSeparateMemoryAddressSpacesCount() + 1;

                  if ( _usePriority ) _queues = NEW SchedQueuesWDPQ( _numMemSpaces );
                  else _queues = NEW SchedQueuesWDQ( _numMemSpaces );

                  if ( _numMemSpaces > 1 ) {
                     _createdData = NEW size_t[_numMemSpaces];
                     for (unsigned int i = 0; i < _numMemSpaces; i += 1 ) {
                        _createdData[i] = 0;
                     }
                  }

                  _load = NEW unsigned int[_numMemSpaces];
                  for (unsigned int i = 0; i < _numMemSpaces; i += 1) {
                     _load[ i ] = 0;
                  }
               }

               ~TeamData ()
               {
                  delete _queues;

                  if (_numMemSpaces > 1 ) {
                     delete[] _createdData;
                     delete[] _load;
                  }
               }

               void addThreadData ( ThreadData * thdata )
               {
                  _teamThrDataLock.acquire();
                  _teamThreadData.insert( thdata );
                  _teamThrDataLock.release();
               }

            };

            /** \brief Cache Scheduler data associated to each thread
              *
              */
            class ThreadData : public ScheduleThreadData
            {
               public:
               /*! queue of ready tasks to be executed */
               bool         _init;
               unsigned int _memId;
               unsigned int _helped;
               unsigned int _fetch;
               PE         * _pe;

               ThreadData () : _init( false ), _memId( 0 ), _helped( 0 ), _fetch( 0 ), _pe( NULL ) {}

               virtual ~ThreadData () {}

               void init ( BaseThread *thread, TeamData &tdata )
               {
                  _pe = thread->runningOn();
                  _memId = _pe->getMemorySpaceId();
                  tdata.addThreadData( this );
                  _init = true;
               }
            };

            template <class CondA, class CondB>
            struct Or {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return CondA::check( wd, thread ) || CondB::check( wd, thread );
               }
            };

            template <class CondA, class CondB>
            struct And {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return CondA::check( wd, thread ) && CondB::check( wd, thread );
               }
            };

            template <class Cond>
            struct Not {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return !Cond::check( wd, thread );
               }
            };

            struct AlreadyInit {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  if ( wd.initialized() ) return true;
                  else return false;
               }
            };

            struct AlreadyDataInit {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  if ( wd.initialized() && wd._mcontrol.isDataReady( wd ) ) return true;
                  else return false;
               }
            };

            struct WouldNotTriggerInvalidation
            {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return wd.resourceCheck( thread, false );
               }
            };

            struct WouldNotRunOutOfMemory
            {
               static inline bool check( WD &wd, BaseThread const &thread ) {
                  return wd.resourceCheck( thread, true );
               }
            };

            struct NoCopy
            {
               static inline bool check( WD &wd, BaseThread const &thread )
               {
                  CopyData * copies = wd.getCopies();
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && copies[i].isInput() ) {
                        NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                        if ( !locs.empty() ) {
                           for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                              if ( ! RegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->first, thread.runningOn()->getMemorySpaceId() ) ) {
                                 return false;
                              }
                           }
                        } else {
                           if ( ! wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( thread.runningOn()->getMemorySpaceId() ) ) {
                              return false;
                           }
                        }
                     }
                  }
                  return true;
               }
            };


            /* disable copy and assigment */
            explicit ReadyCacheSchedPolicy ( const ReadyCacheSchedPolicy & );
            const ReadyCacheSchedPolicy & operator= ( const ReadyCacheSchedPolicy & );


            enum DecisionType { /* 0 */ NOCONSTRAINT,
                                /* 1 */ NOCOPY,
                                /* 2 */ SICOPY,
                                /* 3 */ NOCOPYNOOUTMEM,
                                /* 4 */ SICOPYNOOUTMEM,
                                /* 5 */ ALREADYINIT };

         public:
            static bool _usePriority;
            // Depth propagation inside task graph
            // -1 means no depth limit
            //  0 means no propagation
            static int _priorityPropagation;

            static bool _noSteal;
            static bool _noInvalAware;
            static bool _affinityInout;
            static bool _affinityLoad;


            // constructor
            ReadyCacheSchedPolicy() : SchedulePolicy ( "Ready Cache" ) {
               _usePriority = _usePriority && sys.getPrioritiesNeeded();
            }

            // destructor
            virtual ~ReadyCacheSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               /* Queue 0 will be the global one */
               unsigned int numQueues = 0; // will be computed later

               return NEW TeamData( numQueues );
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
            }

            int computeAffinityScore( WD &wd, unsigned int numNodes, int *scores, size_t &maxPossibleScore );
            void rankWD( BaseThread *thread, WD &wd );

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

               if ( !data._init ) {
                  data.init( thread, tdata );
               }

               if ( tdata._numMemSpaces == 1 ) {
                  tdata._queues->globalPushBack( &wd );
                  return;
               }

               if ( wd.isTied() ) {
                   unsigned int index = wd.isTiedTo()->runningOn()->getMemorySpaceId();
                   tdata._queues->pushBack( &wd, index );
                   return;
               }

               // Check if there is only one memory space where this WD can be run
               unsigned int executors = 0;
               int candidate = -1;

               for ( ThreadDataSet::const_iterator it = tdata._teamThreadData.begin(); it != tdata._teamThreadData.end(); it++ ) {
                  ThreadData * thd = *it;
                  if ( thd->_pe->canRun( wd ) ) {
                     executors++;
                     candidate = thd->_memId;
                  }
               }

               // If we found only one memory space, push this WD to its queue
               if ( executors == 1 ) {
                  tdata._queues->pushBack( &wd, candidate );
                  return;
               }

               tdata._queues->globalPushBack( &wd );
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
               TeamData &tdata = ( TeamData & ) *thread->getTeam()->getScheduleData();

               if ( !data._init ) {
                  data.init( thread, tdata );
               }

               if ( tdata._numMemSpaces == 1 ) {
                  tdata._queues->globalPushBack( &wd );
                  return;
               }

               if ( wd.getNumCopies() > 0 ) {
                  CopyData * copies = wd.getCopies();
                  unsigned int wo_copies = 0, ro_copies = 0, rw_copies = 0;
                  size_t createdDataSize = 0;
                  for (unsigned int idx = 0; idx < wd.getNumCopies(); idx += 1)
                  {
                     if ( !copies[idx].isPrivate() ) {
                        if ( copies[idx].isInput() &&  copies[idx].isOutput() ) {
                           rw_copies += copies[idx].getSize() * 2;
                        } else if ( copies[idx].isInput() ) {
                           ro_copies += copies[idx].getSize();
                        } else if ( copies[idx].isOutput() ) {
                           wo_copies += copies[idx].getSize();
                        }
                        createdDataSize += ( !copies[idx].isInput() && copies[idx].isOutput() ) * copies[idx].getSize();
                     }
                  }

                  if ( rw_copies == 0 ) /* init task */
                  {
                     int winner = -1;
                     for ( ThreadDataSet::const_iterator it = tdata._teamThreadData.begin(); it != tdata._teamThreadData.end(); it++ ) {
                        ThreadData * thd = *it;
                        if ( thd->_pe->canRun( wd ) ) {
                           if ( winner == -1 ) {
                              winner = thd->_memId;
                           } else {
                              winner = ( tdata._createdData[ thd->_memId ] < tdata._createdData[ winner ] ) ? thd->_memId : winner ;
                           }
                        }
                     }

                     tdata._createdData[ winner ] += createdDataSize;
                     tdata._queues->pushBack( &wd, winner );
                  }
                  else
                  {
                        rankWD( thread, wd );
                     //        std::cerr << "END case, regular wd " << wd.getId() << std::endl;
                  }
               } else {
                  // Check which memory spaces this WD can be run on
                  for ( ThreadDataSet::const_iterator it = tdata._teamThreadData.begin(); it != tdata._teamThreadData.end(); it++ ) {
                     ThreadData * thd = *it;
                     if ( thd->_pe->canRun( wd ) ) {
                        tdata._queues->pushBack( &wd, thd->_memId );
                        break;
                     }
                  }
               }
            }

            virtual void atCreate ( DependableObject &depObj )
            {
               if ( _usePriority ) propagatePriority( depObj, _priorityPropagation );
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
               queue( thread,newWD );

               return 0;
            }

            virtual WD *atIdle ( BaseThread *thread, int numSteal );
            virtual WD *atBlock ( BaseThread *thread, WD *current );

            virtual WD *atAfterExit ( BaseThread *thread, WD *current, int  numStealDummy )
            {
               return atBlock(thread, current );
            }

            WD * atPrefetch ( BaseThread *thread, WD &current )
            {
               // Getting the immediate successor notably increases performance
               WD * next = current.getImmediateSuccessor(*thread);

               if ( next != NULL ) return next;

               ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

               if ( !data._init ) {
                  data.init( thread, tdata );
               }

               // Try to schedule the thread with a task from its queue
               next = tdata._queues->popFront( thread, data._memId );

               if ( next != NULL ) return next;

               return atIdle( thread, false );
            }

            WD * atBeforeExit ( BaseThread *thread, WD &current, bool schedule )
            {
               if ( schedule ) {
                  return atPrefetch( thread, current );
               }
               return NULL;
            }

            WD *fetchWD ( BaseThread *thread, WD *current );

            /*!
             * \brief This method performs the main task of the smart priority
             * scheduler, which is to propagate the priority of a WD to its
             * immediate predecessors. It is meant to be invoked from
             * DependenciesDomain::submitWithDependenciesInternal.
             * \param [in/out] predecessor The preceding DependableObject.
             * \param [in] successor DependableObject whose WD priority has to be
             * propagated.
             */
            void atSuccessor ( DependableObject &successor, DependableObject &predecessor )
            {
               if ( !_usePriority || _priorityPropagation == 0 ) return;

               WD *pred = ( WD* ) predecessor.getRelatedObject();
               if ( pred == NULL ) {
                  debug( "AffinityReadyPriority::successorFound predecessor.getRelatedObject() is NULL" )
                  return;
               }

               WD *succ = ( WD* ) successor.getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "AffinityReadyPriority::atSuccessor  successor.getRelatedObject() is NULL" );
               }

               debug ( "Propagating priority from "
                  << ( void * ) succ << ":" << succ->getId() << " to "
                  << ( void * ) pred << ":"<< pred->getId()
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
                  for ( unsigned int i = 0; i < tdata._numMemSpaces; i++ )
                  {
                     tdata._queues->reorderWD( pred, i );
                  }
               }

               // Propagate priority recursively
               propagatePriority( predecessor, _priorityPropagation - 1 );
            }

            void propagatePriority ( DependableObject & successor, int maxDepth )
            {
               if ( maxDepth == 0 ) return;

               WD * succ = ( WD * ) successor.getRelatedObject();
               if ( succ == NULL ) return;

               if ( successor.numPredecessors() == 0 ) return;

               DependableObject::DependableObjectVector & predecessors = successor.getPredecessors();
               for ( DependableObject::DependableObjectVector::iterator it = predecessors.begin();
                     it != predecessors.end(); it++ ) {
                  DependableObject * obj = it->second;
                  WD * pred = ( WD * ) obj->getRelatedObject();
                  if ( pred == NULL ) continue;

                  if ( pred->getPriority() < succ->getPriority() ) {

                     //std::ostringstream str;
                     //str << "REC: Propagating priority from " << succ->getId() << " to " << pred->getId()
                     //   << ", old priority: " << pred->getPriority()
                     //   << ", new priority: " << succ->getPriority() << std::endl;
                     //std::cout << str.str();

                     pred->setPriority( succ->getPriority() );

                     // Reorder
                     TeamData &tdata = ( TeamData & ) *nanos::myThread->getTeam()->getScheduleData();
                     // Do it for all the queues since I don't know which ones have the predecessor
                     // TODO (#652): Find a way to avoid the situation described above.
                     for ( unsigned int i = 0; i < tdata._numMemSpaces; i++ )
                     {
                        tdata._queues->reorderWD( pred, i );
                     }

                     propagatePriority( *obj, maxDepth - 1 );
                  }

                  // Propagate priority
                  //std::ostringstream str2;
                  //str2 << "Calling propagate for " << pred->getId() << std::endl;
                  //std::cout << str2.str();

                  //propagatePriority( obj );
               }
            }

            bool usingPriorities() const
            {
               return _usePriority;
            }
      };

      inline WD *ReadyCacheSchedPolicy::fetchWD( BaseThread *thread, WD *current )
      {
         //WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         if ( !data._init ) {
            data.init( thread, tdata );
         }

         return tdata._queues->fetchWD( thread, data._memId );
      }

      WD *ReadyCacheSchedPolicy::atBlock ( BaseThread *thread, WD *current )
      {
         return atIdle( thread, false );
      }

      /*!
       */
      WD * ReadyCacheSchedPolicy::atIdle ( BaseThread *thread, int numSteal )
      {
         WorkDescriptor * wd = NULL;

         ThreadData &data = ( ThreadData & ) *thread->getTeamData()->getScheduleData();
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         if ( !data._init ) {
            data.init( thread, tdata );
         }

         if ( tdata._numMemSpaces == 1 ) {
            wd = tdata._queues->globalPopFront( thread );
            return wd;
         }

         wd = fetchWD( thread, NULL );

         if ( wd != NULL ) return wd;

         /*
          * Try to get it from the global queue and assign it properly
          */
          wd = tdata._queues->globalPopFront( thread );

          if ( wd != NULL ) {
             affinity_queue( thread, *wd );
             wd = fetchWD( thread, NULL );

             if ( wd != NULL ) return wd;
          }

          if ( !_noSteal )
          {
             for ( unsigned int i = data._memId + 1; i < tdata._numMemSpaces; i++ ) {
                if ( tdata._queues->size( i ) > 1 ) {
                   wd = tdata._queues->popBack( thread, i );
                   return wd;
                }
             }
             for ( unsigned int i = 0; i < data._memId; i++ ) {
                if ( tdata._queues->size( i ) > 1 ) {
                   wd = tdata._queues->popBack( thread, i );
                   return wd;
                }
             }
          }

         if ( wd == NULL ) {
            OS::nanosleep( 100 );
         }

         return wd;
      }

      int ReadyCacheSchedPolicy::computeAffinityScore( WD &wd, unsigned int numMemSpaces, int *scores, size_t &maxPossibleScore )
      {
         CopyData * copies = wd.getCopies();

         maxPossibleScore = 0;
         for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
            if ( !copies[i].isPrivate() && (
                  ( !_affinityInout ) || ( copies[i].isInput() && copies[i].isOutput() && _affinityInout )
            ) ) {
               NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
               maxPossibleScore += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
               if ( locs.empty() ) {
                  // Data not fragmented between different memory spaces
                  for ( unsigned int mem = 0; mem < numMemSpaces; mem++ ) {
                     if ( scores[mem] != -1 ) {
                        if ( wd._mcontrol._memCacheCopies[ i ]._reg.isLocatedIn( mem ) ) {
                           scores[ mem ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                        }
                     }
                  }
               } else {
                  for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                     for ( unsigned int mem = 0; mem < numMemSpaces; mem++ ) {
                        if ( scores[mem] != -1 ) {
                           if ( RegionDirectory::isLocatedIn( wd._mcontrol._memCacheCopies[ i ]._reg.key, it->second, mem ) ) {
                              scores[ mem ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                           }
                        }
                     }
                  }
               }
            } //else { std::cerr << "ignored copy "<< std::endl; }
         }

         int winner = -1;
         unsigned int start = 0;
         for ( unsigned int mem = 0; mem < numMemSpaces; mem++ ) {
            if ( scores[mem] != -1 ) {
               start = mem;
               break;
            }
         }

         int maxRank = -1;
         for ( unsigned int i = start; i < numMemSpaces; i++ ) {
            if ( scores[i] > maxRank ) {
               winner = i;
               maxRank = scores[i];
            }
         }

         if ( winner == -1 )
            winner = start;
         return winner;
      }

      void ReadyCacheSchedPolicy::rankWD( BaseThread *thread, WD &wd )
      {
         TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

         int scores[ tdata._numMemSpaces ];

         for ( unsigned int i = 0; i < tdata._numMemSpaces; i++ ) {
            scores[i] = -1;
         }

         for ( ThreadDataSet::const_iterator it = tdata._teamThreadData.begin(); it != tdata._teamThreadData.end(); it++ ) {
            ThreadData * thd = *it;
            if ( scores[ thd->_memId ] == -1 && thd->_pe->canRun( wd ) ) {
               scores[ thd->_memId ] = 0;
            }
         }

         //std::cerr << "RANKING WD " << wd.getId() << " numCopies " << wd.getNumCopies() << std::endl;
         size_t max_possible_score = 0;
         int winner = computeAffinityScore( wd, tdata._numMemSpaces, scores, max_possible_score );

         if ( _affinityLoad ) {
            unsigned int usage[ tdata._numMemSpaces ];
            unsigned int ties = 0;
            int maxRank = scores[ winner ];
            unsigned int start = 0;

            for ( unsigned int i = start; i < tdata._numMemSpaces; i++ ) {
               //std::cerr << "winner is "<< winner << " ties "<< ties << " " << maxRank<< " this score "<< scores[i] << std::endl;
               if ( scores[i] == maxRank ) {
                  usage[ ties ] = i;
                  ties += 1;
               }
            }
            //std::cerr << "winner is "<< winner << " ties "<< ties << " " << maxRank<< std::endl;
            if ( ties > 1 ) {
               //std::cerr << "Max score is " << maxRank << " / " << max_possible_score << ", I have to chose between :";
               //for ( unsigned int ii = 0; ii < ties; ii += 1 ) fprintf(stderr, " %d", usage[ ii ] );
               //std::cerr << std::endl;
               unsigned int minLoad = usage[0];
               for ( unsigned int ii = 1; ii < ties; ii += 1 ) {
                  //     std::cerr << "load of (min) " << minLoad << " is " << tdata._load[ minLoad ] <<std::endl;
                  //   std::cerr << "load of (itr) " << usage[ ii ]  << " is " << tdata._load[ usage[ ii ] ] << std::endl;
                  if ( tdata._queues->size( usage[ ii ] ) < tdata._queues->size( minLoad ) ) {
                     minLoad = usage[ ii ];
                  }
               }
               //std::cerr << "Well winner is gonna be "<< minLoad << std::endl;
               winner = minLoad;
            }
         }

         wd._mcontrol.setAffinityScore( scores[ winner ] );
         wd._mcontrol.setMaxAffinityScore( max_possible_score );

         /* end of rank by memory space */

         tdata._queues->pushBack( &wd, winner );
      }

      bool ReadyCacheSchedPolicy::_usePriority = true;
      int ReadyCacheSchedPolicy::_priorityPropagation = 5;
      bool ReadyCacheSchedPolicy::_noSteal = false;
      bool ReadyCacheSchedPolicy::_noInvalAware = false;
      bool ReadyCacheSchedPolicy::_affinityInout = false;
      bool ReadyCacheSchedPolicy::_affinityLoad = false;


      class ReadyCacheSchedPlugin : public Plugin
      {
         public:
            ReadyCacheSchedPlugin() : Plugin( "Cache-guided scheduling Plugin for ready tasks",1 ) {}

            virtual void config( Config& cfg )
            {
               cfg.setOptionsSection( "Ready-Affinity module", "Data Affinity scheduling module at ready task time" );

               cfg.registerConfigOption ( "affinity-priority", NEW Config::FlagOption( ReadyCacheSchedPolicy::_usePriority ), "Priority queue used as ready task queue");
               cfg.registerArgOption( "affinity-priority", "affinity-priority" );

               cfg.registerConfigOption ( "affinity-priority-depth", NEW Config::IntegerVar( ReadyCacheSchedPolicy::_priorityPropagation ), "Number of levels to propagate priority upwards in the task graph (0 = no propagation, -1 = no depth limit)");
               cfg.registerArgOption( "affinity-priority-depth", "affinity-priority-depth" );

               cfg.registerConfigOption ( "affinity-no-steal", NEW Config::FlagOption( ReadyCacheSchedPolicy::_noSteal ), "Steal tasks from other threads");
               cfg.registerArgOption( "affinity-no-steal", "affinity-no-steal" );

               cfg.registerConfigOption ( "affinity-no-inval-aware", NEW Config::FlagOption( ReadyCacheSchedPolicy::_noInvalAware ), "Do not take into account invalidations");
               cfg.registerArgOption( "affinity-no-inval-aware", "affinity-no-inval-aware" );

               cfg.registerConfigOption ( "affinity-inout", NEW Config::FlagOption( ReadyCacheSchedPolicy::_affinityInout ), "Check affinity for inout data only");
               cfg.registerArgOption( "affinity-inout", "affinity-inout" );

               cfg.registerConfigOption ( "affinity-load", NEW Config::FlagOption( ReadyCacheSchedPolicy::_affinityLoad ), "Also take into account system load and try to balance work assignment");
               cfg.registerArgOption( "affinity-load", "affinity-load" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(NEW ReadyCacheSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-affinity-ready",nanos::ext::ReadyCacheSchedPlugin);
