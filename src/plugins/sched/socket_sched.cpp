/*************************************************************************************/
/*      Copyright 2012 Barcelona Supercomputing Center                               */
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
#include <cmath>
#include <fstream>
#include <sstream>
#ifdef HWLOC
#include <hwloc.h>
#endif

namespace nanos {
   namespace ext {

      class SocketSchedPolicy : public SchedulePolicy
      {
         public:
         
         private:
            /*! \brief Steal work from other sockets? */
            bool _steal;
            /*! \brief Use immediate successor when prefecting. */
            bool _useSuccessor;
            /*! \brief Use smart priority (propagate priority) */
            bool _smartPriority;
            /*! \brief Number of loops to spin before attempting stealing */
            unsigned _spins;
            
            /*! \brief For a given socket, a list of near sockets. */
            typedef std::vector<unsigned> NearSocketsList;
            
            /*! \brief Info related to socket distance and stealing */
            struct SocketDistanceInfo {
               NearSocketsList list;
               
               //! Next queue to steal from (round robin)
               Atomic<unsigned> stealNext;
            };
            
            /*! \brief For each socket, an array of close sockets. */
            typedef std::vector<SocketDistanceInfo> NearSocketsMatrix;
            
            /*! \brief Keep a vector with all the close sockets for every socket */
            NearSocketsMatrix _nearSockets;

            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue*           _readyQueues;
               //! Next queue to insert to (round robin scheduling)
               // TODO(gmiranda): remove this since we don't use it
               Atomic<unsigned>           _next;
               //! If there is an active "master" thread, for every socket
               Atomic<bool>*               _activeMasters;
 
               TeamData ( unsigned int sockets ) : ScheduleTeamData(), _next( 0 )
               {
                  _readyQueues = NEW WDPriorityQueue[ sockets*2 + 1 ];
                  _activeMasters = NEW Atomic<bool>[ sockets ];
               }

               ~TeamData () {
                  delete[] _readyQueues;
                  delete[] _activeMasters;
               }
            };

            /** \brief Socket Scheduler data associated to each thread
              *
              */
            struct ThreadData : public ScheduleThreadData
            {
               /*! queue of ready tasks to be executed */
               unsigned int _cacheId;
               bool _init;

               ThreadData () : _cacheId(0), _init(false) {}
               virtual ~ThreadData () {
               }
            };
         
            /** \brief Comparison functor used in distance computations.
             */
            struct DistanceCmp{
               unsigned* _distances;
               
               DistanceCmp( unsigned* distances )
                  : _distances( distances ){}
               
               bool operator()( unsigned socket1, unsigned socket2 )
               {
                  return _distances[socket1] < _distances[socket2];
               }
            };

            /* disable copy and assigment */
            explicit SocketSchedPolicy ( const SocketSchedPolicy & );
            const SocketSchedPolicy & operator= ( const SocketSchedPolicy & );
         
         private:
            /** \brief Creates lists of close sockets.
             */
            void computeDistanceInfo() {
               // Fill the distance info matrix
               _nearSockets.resize( sys.getNumSockets() );
               
               // For every numa node
               for ( unsigned from = 0; from < _nearSockets.size(); ++from )
               {
                  _nearSockets[ from ].stealNext = 0;
                  NearSocketsList& row = _nearSockets[ from ].list;
                  row.reserve( sys.getNumSockets() - 1 );
                  
                  unsigned distances[ _nearSockets.size() ];
                  std::stringstream path;
                  path << "/sys/devices/system/node/node";
                  path << from << "/distance";
                  std::ifstream fDistances( path.str().c_str() );
                  
                  
                  std::copy (std::istream_iterator<unsigned>( fDistances ),
                     std::istream_iterator<unsigned>(),
                     distances
                  );
                  
                  fDistances.close();
                  
                  for ( unsigned to = 0; to < (unsigned) sys.getNumSockets(); ++to )
                  {
                     //fprintf( stderr, "Distance from %d to %d: %d\n", from, to, distances[to] );
                     if ( to == from )
                        continue;
                     row.push_back( to );
                  }
                  if ( !row.empty() ) {
                     // Sort by distance
                     std::sort( row.begin(), row.end(), DistanceCmp( distances ) );
                     // Keep only close nodes
                     NearSocketsList::iterator it = std::upper_bound(
                        row.begin(), row.end(), row.front(),
                        DistanceCmp( distances )
                     );
                     row.erase( it, row.end() );
                  }
               }
            }

         public:
            // constructor
            SocketSchedPolicy ( bool steal, bool useSuccessor, bool smartPriority,
               unsigned spins )
               : SchedulePolicy ( "Socket" ), _steal( steal ),
               _useSuccessor( useSuccessor ), _smartPriority( smartPriority ),
               _spins ( spins )
            {
               int numSockets = sys.getNumSockets();
               int coresPerSocket = sys.getCoresPerSocket();
               
               //fprintf( stderr, "Steal: %d, successor: %d, smart: %d, spins: %d\n", steal, useSuccessor, smartPriority, spins );

               // Check config
               if ( numSockets != std::ceil( sys.getNumPEs() / static_cast<float>( coresPerSocket) ) )
               {
                  unsigned validSockets = std::ceil( sys.getNumPEs() / static_cast<float>(coresPerSocket) );
                  warning0( "Adjusting num-sockets from " << numSockets << " to " << validSockets );
                  sys.setNumSockets( validSockets );
               }
               
               computeDistanceInfo();
            }

            // destructor
            virtual ~SocketSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               // Create 2 queues per socket plus one for the global queue.
               return NEW TeamData( sys.getNumSockets() );
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
            }

            virtual void queue ( BaseThread *thread, WD &wd )
            {
               socketQueue( thread, wd, false );
            }
            
            /*!
             *  \brief Queues a work descriptor in a readyQueue.
             *  It will reuse the queue the wd was previously in.
             *  \note Don't call this from atSubmit, otherwise all WDs will be
             *  sent to the default queue (0).
             *  \param thread pointer to the thread to which readyQueue the task must be appended
             *  \param wd a reference to the work descriptor to be enqueued
             *  \see distribute
             *  \sa ThreadData, WD and BaseThread
             */
            void socketQueue ( BaseThread *thread, WD &wd, bool wakeUp )
            {
               unsigned index = wd.getWakeUpQueue();
               // FIXME: use another variable to check this condition
               // If the WD has not been distributed yet, distribute it
               if ( index == UINT_MAX )
                  return distribute( thread, wd );
               
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               
               switch( wd.getDepth() ) {
                  case 0:
                     //fprintf( stderr, "Wake up Depth 0, inserting WD %d in queue number 0\n", wd.getId() );
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  // Keep other tasks in the same socket as they were
                  // Note: we might want to insert the ones with depth 1 in the front
                  case 1:
                     //fprintf( stderr, "Wake up Depth 1, inserting WD %d in queue number %d\n", wd.getId(), index );
                     
                     if ( wakeUp )
                        tdata._readyQueues[index].push_front ( &wd );
                     else
                        tdata._readyQueues[index].push_back ( &wd );
                     break;
                  default:
                     //fprintf( stderr, "Wake up Depth >1, inserting WD %d in queue number %d\n", wd.getId(), index );
                     
                     // Insert at the back
                     tdata._readyQueues[index].push_back ( &wd );
                     break;
               }
            }
            
            /*!
             *  \brief Queues a new work descriptor in a readyQueue.
             *  It will insert WDs with depth 1 following a round-robin schedule
             *  policy. WDs with depth > 1 will be sent to the queue of the
             *  thread's socket.
             *  \note Don't call this from atWakeUp, or tasks with depth 1 can
             *  be inserted in a different queue than the one they were
             *  previously in.
             *  \param thread pointer to the thread to which readyQueue the task must be appended
             *  \param wd a reference to the work descriptor to be enqueued
             *  \see queue
             */
            virtual void distribute ( BaseThread *thread, WD &wd )
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               if( wd.getWakeUpQueue() != UINT_MAX )
                  warning0( "WD already has a queue (" << wd.getWakeUpQueue() << ")" );
               
               unsigned index;
               
               switch( wd.getDepth() ) {
                  case 0:
                     //fprintf( stderr, "Depth 0, inserting WD %d in queue number 0\n", wd.getId() );
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  case 1:
                     //index = (tdata._next++ ) % sys.getNumSockets() + 1;
                     // 2 queues per socket, the first one is for level 1 tasks
                     index = (wd._socket % sys.getNumSockets())*2 + 1;
                     wd.setWakeUpQueue( index );
                     
                     //fprintf( stderr, "Depth 1, inserting WD %d in queue number %d (curr socket %d)\n", wd.getId(), index, wd._socket );
                     
                     // Insert at the front (these will have higher priority)
                     tdata._readyQueues[index].push_back ( &wd );
                     
                     // Round robin
                     //tdata._next = ( socket+1 ) % sys.getNumSockets();
                     //fprintf( stderr, "Next = %d\n", tdata._next.value() );
                     break;
                  default:
                     // Insert this in its parent's socket
                     index = wd.getParent()->getWakeUpQueue();
                     // If index is not even
                     if ( index % 2 != 0 )
                        // Means its parent is level 1, small tasks go in even queues
                        ++index;
                     
                     wd.setWakeUpQueue( index );
                     
                     //fprintf( stderr, "Depth %d>1, inserting WD %d in queue number %d\n", wd.getDepth(), wd.getId(), index );
                     // Insert at the back
                     tdata._readyQueues[index].push_back ( &wd );
                     break;
               }
            }

            /*!
             *  \brief Function called when a new task must be created.
             *  \param thread pointer to the thread to which belongs the new task
             *  \param wd a reference to the work descriptor of the new task
             *  \sa WD and BaseThread
             */
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               distribute( thread, newWD );

               return 0;
            }

            virtual WD * atIdle ( BaseThread *thread )
            {
               WD* wd = NULL;
               
               // Get the socket of this thread
               unsigned socket = thread->getSocket();
               
               //fprintf( stderr, "atIdle socket %d\n", socket );
               
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               
               /*
                * Which queue should we look at?
                * If the higher depth tasks queue has < N (N=X*cores_per_socket)
                * tasks, and there's no other thread in this socket doing
                * depth 1 tasks, query the depth 1 queue.
                * TODO: compute N.
                * TODO: just one thread at a time can run depth 1 tasks.
                */
               int deepTasksN = tdata._readyQueues[socket*2+2].size();
               bool emptyBigTasks = tdata._readyQueues[socket*2+1].empty();
               //fprintf( stderr, "[sockets] %d tasks at the small's queue\n", deepTasksN );
               
               //unsigned thId = thread->getId();
               
               unsigned queueNumber;
               // TODO Improve atomic condition
               if ( deepTasksN < 1*sys.getCoresPerSocket() && !emptyBigTasks
                   /*&& ( tdata._activeMasters[socket].value() == 0 || tdata._activeMasters[socket].value() == thId )*/ )
               {
                  //tdata._activeMasters[socket] = thId + 1;
                  queueNumber = socket*2+1;
               }
               else
                  queueNumber = socket*2+2;
               
               unsigned spins = _spins;
               // Make sure the queue is really empty... lotsa times!
               do {
                  // We only spin when steal is enabled
                  wd = tdata._readyQueues[queueNumber].pop_front( thread );
                  --spins;
               } while( _steal && wd == NULL && spins != 0 );
               
               if ( wd != NULL )
                  return wd;
               
               // If we want/need to steal
               if ( _steal )
               {
                  if ( false /* steal from the biggest */ )
                  {
                     // Find the queue with the most small tasks
                     WDPriorityQueue *largest = &tdata._readyQueues[2];
                     int largestSocket = 0;
                     for ( int i = 1; i < sys.getNumSockets(); ++i )
                     {
                        WDPriorityQueue *current = &tdata._readyQueues[ (i+1)*2];
                        if ( largest->size() < current->size() ){
                           largest = current;
                           largestSocket = i;
                        }
                     }
                     //fprintf( stderr, "Stealing from socket #%d that has %lu tasks\n", largestSocket, largest->size() );
                     wd = largest->pop_front( thread );
                  }
                  // Round robbin steal
                  else {
                     unsigned closeIndex = _nearSockets[socket].stealNext.value();
                     _nearSockets[socket].stealNext = ++_nearSockets[socket].stealNext % ( sys.getNumSockets() - 1 );
                     unsigned close = _nearSockets[socket].list[ closeIndex ];
                     
                     // 2 queues per socket + 1 master queue + 1 (offset of the inner tasks)
                     unsigned index = close * 2 + 2;
                     //fprintf( stderr, "Stealing from index: %d\n", index );
                     wd = tdata._readyQueues[index].pop_front( thread );
                  }
                  
                  if ( wd != NULL )
                     return wd;
               }
               
               // If this queue is empty, try the global queue
               return tdata._readyQueues[0].pop_front( thread );
            }
            
            virtual WD * atWakeUp( BaseThread *thread, WD &wd )
            {
               // If the WD was waiting for something
               if ( wd.started() ) {
                  BaseThread * prefetchThread = NULL;
                  // Check constraints since they won't be checked in Schedule::wakeUp
                  if ( Scheduler::checkBasicConstraints ( wd, *thread ) ) {
                     prefetchThread = thread;
                  }
                  else
                     prefetchThread = wd.isTiedTo();
                  
                  // Returning the wd here makes the application to hang
                  // Use prefetching instead.
                  if ( prefetchThread != NULL && prefetchThread->reserveNextWD() ) {
                     prefetchThread->setReservedNextWD( &wd );
                     
                     return NULL;
                  }
               }
               
               // otherwise, as usual
               socketQueue( thread, wd, false );
               
               return NULL;
            }
            
            WD * atPrefetch ( BaseThread *thread, WD &current )
            {
               // If the use of getImmediateSuccessor is not enabled
               if ( !_useSuccessor )
                  // Revert to the base behaviour
                  return SchedulePolicy::atPrefetch( thread, current );
               
               WD * found = current.getImmediateSuccessor(*thread);
            
               return found != NULL ? found : atIdle(thread);
            }
         
            //WD * atBeforeExit ( BaseThread *thread, WD &current )
            //{
            //   // If the use of getImmediateSuccessor is not enabled
            //   if ( !_useSuccessor )
            //      // Revert to the base behaviour
            //      return SchedulePolicy::atBeforeExit( thread, current );
            //   
            //   return current.getImmediateSuccessor(*thread);
            //}

            
            /*virtual WD * atBlock( BaseThread *thread, WD *current )
            {
               return atWakeUp( thread, *current );
            }*/
            
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
               if ( !_smartPriority )
                  return;
               
               debug( "SmartPriority::successorFound" );
               if ( predecessor == NULL ) {
                  debug( "SmartPriority::successorFound predecessor is NULL" );
                  return;
               }
               if ( successor == NULL ) {
                  debug( "SmartPriority::successorFound successor is NULL" );
                  return;
               }
               
               WD *pred = ( WD* ) predecessor->getRelatedObject();
               if ( pred == NULL ) {
                  debug( "SmartPriority::successorFound predecessor->getRelatedObject() is NULL" )
                  return;
               }
               
               WD *succ = ( WD* ) successor->getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "SmartPriority::successorFound  successor->getRelatedObject() is NULL" );
               }
               
               //debug( "Predecessor[" << pred->getId() << "]" << pred << ", Successor[" << succ->getId() << "]" );
               
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
                  unsigned index = pred->getWakeUpQueue();
                  // What happens if pred is not in any queue? Fatal.
                  if ( index < static_cast<unsigned>( sys.getNumSockets() ) )
                     tdata._readyQueues[ index ].reorderWD( pred );
               }
            }
      };

      class SocketSchedPlugin : public Plugin
      {
         private:
            int _numSockets;
            int _coresPerSocket;
            
            bool _steal;
            bool _immediate;
            bool _smart;
            
            unsigned _spins;
            
            void loadDefaultValues()
            {
               // Read number of sockets and cores from hwloc
#ifdef HWLOC
               hwloc_topology_t topology;
               
               /* Allocate and initialize topology object. */
               hwloc_topology_init( &topology );
               
               /* Perform the topology detection. */
               hwloc_topology_load( topology );
               int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_NODE );
               
               if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
                  _numSockets = hwloc_get_nbobjs_by_depth(topology, depth);
               }
               
               depth = hwloc_get_type_depth( topology, HWLOC_OBJ_CORE );
               if ( depth != HWLOC_TYPE_DEPTH_UNKNOWN ) {
                  _coresPerSocket = hwloc_get_nbobjs_by_depth( topology, depth ) / _numSockets;
               }
               
               /* Destroy topology object. */
               hwloc_topology_destroy(topology);
#else
               // Number of sockets can be read with
               // cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
               // Cores per socket:
               // cat /proc/cpuinfo | grep 'core id' | sort | uniq | wc -l
               debug0( "No hwloc support" );
               _numSockets = sys.getNumSockets();
               _coresPerSocket = sys.getCoresPerSocket();
#endif
            }
         public:
            SocketSchedPlugin() : Plugin( "Socket-aware scheduling Plugin",1 ),
               _steal( true ), _immediate( false ), _smart( false ), _spins( 200 ) {}

            virtual void config( Config& cfg ) {
               // Read hwloc's info before reading user parameters
               loadDefaultValues();
               
               cfg.setOptionsSection( "Sockets module", "Socket-aware scheduling module" );
   
               cfg.registerConfigOption( "cores-per-socket", NEW Config::PositiveVar( _coresPerSocket ), "Number of cores per socket." );
               cfg.registerArgOption( "cores-per-socket", "cores-per-socket" );
               
               cfg.registerConfigOption( "num-sockets", NEW Config::PositiveVar( _numSockets ), "Number of sockets available." );
               cfg.registerArgOption( "num-sockets", "num-sockets" );
               
               cfg.registerConfigOption( "socket-steal", NEW Config::FlagOption( _steal ), "Enable work stealing from other sockets' inner tasks queues (default)." );
               cfg.registerArgOption( "socket-steal", "socket-steal" );
               
               cfg.registerConfigOption( "socket-immediate", NEW Config::FlagOption( _immediate ), "Use the immediate successor when prefecting (disabled by default)." );
               cfg.registerArgOption( "socket-immediate", "socket-immediate" );
               
               cfg.registerConfigOption( "socket-smartpriority", NEW Config::FlagOption( _smart ), "Propagates priority to the immediate predecessors (disabled by default)." );
               cfg.registerArgOption( "socket-smartpriority", "socket-smartpriority" );
               
               cfg.registerConfigOption( "socket-steal-spin", NEW Config::UintVar( _spins ), "Number of spins before stealing (200 by default)." );
               cfg.registerArgOption( "socket-steal-spin", "socket-steal-spin" );
            }

            virtual void init() {
               //fprintf(stderr, "Setting numSockets to %d and coresPerSocket to %d\n", _numSockets, _coresPerSocket );
               sys.setNumSockets( _numSockets );
               sys.setCoresPerSocket( _coresPerSocket );
               
               sys.setDefaultSchedulePolicy( NEW SocketSchedPolicy( _steal, _immediate, _smart, _spins ) );
            }
      };

   }
}

DECLARE_PLUGIN("sched-socket",nanos::ext::SocketSchedPlugin);
