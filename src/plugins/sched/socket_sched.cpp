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
#ifdef GPU_DEV
#include "gpudd.hpp"
#endif

namespace nanos {
   namespace ext {

      class SocketSchedPolicy : public SchedulePolicy
      {
         public:
         
         private:
            /*! \brief Steal work from other sockets? */
            bool _steal;
            /*! \brief Steal top level tasks (true) or children (false) */
            bool _stealParents;
            /*! \brief Steal low priority tasks (true) or high (false) */
            bool _stealLowPriority;
            /*! \brief Use immediate successor when prefecting. */
            bool _useSuccessor;
            /*! \brief Use smart priority (propagate priority) */
            bool _smartPriority;
            /*! \brief Number of loops to spin before attempting stealing */
            unsigned _spins;
            /*! \brief If enabled, steal from random queues, otherwise round robin */
            bool _randomSteal;
            
            /*! \brief For a given socket, a list of near sockets. */
            typedef std::vector<unsigned> NearSocketsList;
            
            /*! \brief Info related to socket distance and stealing */
            struct SocketDistanceInfo {
               NearSocketsList list;
               
               // TODO: Change stealNext by just next...
               //! Next index of the list to steal from (round robin)
               Atomic<unsigned> stealNext;
               
               /*! \brief Returns the socket to steal from */
               unsigned getStealNext()
               {
                  // FIXME: Is this mod ok?
                  //fprintf( stderr, "sys.getNumSockets: %d - 1 == list.size() %u?\n", sys.getNumSockets(), (unsigned)list.size() );
                  //fatal_cond( (unsigned)( sys.getNumSockets() - 1 ) != list.size(), "Wrong size of the node list" );
                  unsigned closeIndex = ( ++stealNext ) % list.size();
                  //unsigned close = _nearSockets[socket].list[ closeIndex ];
                  //fprintf( stderr, "Close: %d\n", close );
                  return list[ closeIndex ];
               }
            };
            
            /*! \brief For each socket, an array of close sockets. */
            typedef std::vector<SocketDistanceInfo> NearSocketsMatrix;
            
            /*! \brief Keep a vector with all the close sockets for every socket */
            NearSocketsMatrix _nearSockets;
            
            /*! \brief A set of all the nodes with GPUs */
            std::set<int> _gpuNodes;
            
            /*! \brief List of gpu nodes that will be used to give away work
             * descriptors in round robin.
             */
            SocketDistanceInfo _gpuNodesToGive;

            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue<>*         _readyQueues;
               //! Next queue to insert to (round robin scheduling)
               // TODO(gmiranda): remove this since we don't use it
               Atomic<unsigned>           _next;
               //! If there is an active "master" thread, for every socket
               Atomic<bool>*              _activeMasters;
 
               TeamData ( unsigned int sockets ) : ScheduleTeamData(), _next( 0 )
               {
                  //fprintf( stderr, "Reserving %d queues\n", sockets*2 + 1 );
                  _readyQueues = NEW WDPriorityQueue<>[ sockets*2 + 1 ];
                  // Print how many things are in here.
                  //for( unsigned i = 0; i < ( sockets*2 + 1 ); ++i )
                     //fprintf( stderr, "%u tasks in the queue %i\n", (unsigned) _readyQueues[i].size(), i );
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
                  
                  if ( fDistances.good() ) {
                     std::copy (std::istream_iterator<unsigned>( fDistances ),
                        std::istream_iterator<unsigned>(),
                        distances
                     );
                  }
                  
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
            
            /*!
             *  \brief Finds all the nodes with GPUs. Constructs a set of nodes
             *  that do have GPUs, so it can be queried later on.
             */
            void findGPUNodes()
            {
               // Look for GPUs in all the nodes
               for ( int i = 0; i < sys.getNumWorkers(); ++i )
               {
                  BaseThread *worker = sys.getWorker( i );
#ifdef GPU_DEV
                  //if ( dynamic_cast<GPUDevice*>( worker->runningOn()->getDeviceType() ) == 0 )
                  if ( nanos::ext::GPU == worker->runningOn()->getDeviceType() )
                  {
                     int node = worker->runningOn()->getNUMANode();
                     // Convert to virtual
                     int vNode = sys.getVirtualNUMANode( node );
                     _gpuNodes.insert( vNode );
                     verbose0( "Found GPU Worker in node " << node << " (virtual " << vNode << ")" );
                  }
#endif
                  // Avoid unused variable warning.
                  worker = worker;
               }
               // Initialise the structure that will cycle through gpu nodes
               // when giving away GPU tasks
               _gpuNodesToGive.stealNext = 0;
               // Copy the set elements to a list (since it is easier to cycle
               // through)
               std::copy( _gpuNodes.begin(), _gpuNodes.end(),
                  std::back_inserter( _gpuNodesToGive.list ) );
            }
            
            /*!
             *  \brief Checks if a wd can be run in a node.
             *  If a GPU task is being scheduled in a node with no GPU threads,
             *  it will return false.
             */
            inline bool canRunInNode( WD& wd, int node )
            {
#ifdef GPU_DEV
               // If it's a GPU wd and the node has GPUs, it can run
               if ( wd.canRunIn( nanos::ext::GPU ) ) {
                  //fprintf( stderr, "GPU WD, can it run in node %d? %d\n", node, (int)_gpuNodes.count( node ) );
                  return _gpuNodes.count( node ) != 0;
               }
#endif
               // FIXME: Otherwise assume it's SMP and can always run in this node.
               // This might not be always true.
               return true;
            }
            
            inline int findBetterNode( WD& wd, int node )
            {
               //return node;
            //#if 0
               /* We do not use round robbin here, we just send it to the
                * closest node */
               //int newNode = _nearSockets[ node ].list.front();
               int newNode = _gpuNodesToGive.getStealNext();
               fatal_cond( _gpuNodes.count( newNode ) == 0, "Cannot find a node with GPUs to move the task to." );
               
               
               verbose( "WD " << wd.getId() << " cannot run in node " << node << ", moved to " << newNode );
               return newNode;
            //#endif
            }
            
            /*!
             *  \brief Converts from a node number to a queue index.
             *  \param node Node number (from 0..N-1 nodes).
             *  \param topLevel If the queue for top level task is desired.
             *  Otherwise, the queue for child task will be returned.
             *  \return Queue index (from 0..N*2+1).
             */
            inline unsigned nodeToQueue( unsigned node, bool topLevel ) const
            {
               if ( topLevel ) {
                  return node * 2 + 1;
               }
               // Child task queue
               return node * 2 + 2;
            }
            
            /*!
             *  \brief Converts from a queue number to a node number.
             *  Note that there are 2 queues per node, and that the first queue
             *  can run in all nodes.
             *  \param index Queue index. Must be greater than 0.
             *  \return The node that the queue corresponds to.
             */
            inline unsigned queueToNode( unsigned index ) const
            {
               fatal_cond( index == 0, "Cannot convert to node number the queue index 0" );
               return ( index - 1 ) / 2;
            }

         public:
            // constructor
            SocketSchedPolicy ( bool steal, bool stealParents, bool stealLowPriority,
               bool useSuccessor, bool smartPriority,
               unsigned spins, bool randomSteal )
               : SchedulePolicy ( "Socket" ), _steal( steal ),
               _stealParents( stealParents ), _stealLowPriority( stealLowPriority ),
               _useSuccessor( useSuccessor ), _smartPriority( smartPriority ),
               _spins ( spins ), _randomSteal( randomSteal )
            {
               //int numSockets = sys.getNumSockets();
               //int coresPerSocket = sys.getCoresPerSocket();
               
               //fprintf( stderr, "Steal: %d, successor: %d, smart: %d, spins: %d\n", steal, useSuccessor, smartPriority, spins );

               if ( steal && sys.getNumSockets() == 1 )
               {
                  fatal0( "Steal can not be enabled with just one socket" );
               }
               
               // Check config
               // gmiranda: disabled temporary since NumPES < numWorkers when using GPUS
               //if ( numSockets != std::ceil( sys.getNumPEs() / static_cast<float>( coresPerSocket) ) )
               //{
               //   unsigned validSockets = std::ceil( sys.getNumPEs() / static_cast<float>(coresPerSocket) );
               //   warning0( "Adjusting num-sockets from " << numSockets << " to " << validSockets );
               //   sys.setNumSockets( validSockets );
               //}
               
            }

            // destructor
            virtual ~SocketSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               // Now we can find GPU nodes since the workers will have been created
               findGPUNodes();
               
               computeDistanceInfo();
               
               // Create 2 queues per socket plus one for the global queue.
               return NEW TeamData( sys.getNumAvailSockets() );
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
               int node;
               
               switch( wd.getDepth() ) {
                  case 0:
                     //fprintf( stderr, "Depth 0, inserting WD %d in queue number 0\n", wd.getId() );
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  case 1:
                     node = wd.getSocket();
                     // If the node cannot execute this WD
                     if ( !canRunInNode( wd, node ) )
                        node = findBetterNode( wd, node );
                     
                     //index = (tdata._next++ ) % sys.getNumSockets() + 1;
                     // 2 queues per socket, the first one is for level 1 tasks
                     fatal_cond( node >= sys.getNumAvailSockets(), "Invalid node selected" );
                     //index = (node % sys.getNumSockets())*2 + 1;
                     index = nodeToQueue( node, true );
                     wd.setWakeUpQueue( index );
                     
                     //fprintf( stderr, "Depth 1, inserting WD %d in queue number %d (curr socket %d)\n", wd.getId(), index, wd.runningOn()->getNUMANode() );
                     
                     // Insert at the front (these will have higher priority)
                     tdata._readyQueues[index].push_back ( &wd );
                     
                     // Round robin
                     //tdata._next = ( socket+1 ) % sys.getNumSockets();
                     //fprintf( stderr, "Next = %d\n", tdata._next.value() );
                     break;
                  default:
                     // Insert this in its parent's socket
                     index = wd.getParent()->getWakeUpQueue();
                     
                     node = queueToNode( index );
                     // If this wd cannot run in this node
                     if ( !canRunInNode( wd, node ) ) {
                        node = findBetterNode( wd, node );
                        fatal_cond( node >= sys.getNumAvailSockets(), "Invalid node selected" );
                        // If index is not even
                        // Means its parent is level 1, small tasks go in even queues
                        index = nodeToQueue( node, index % 2 != 0);
                     }
                     else
                     {
                        // If index is not even
                        // Means its parent is level 1, small tasks go in even queues
                        if ( index % 2 != 0 )
                           ++index;
                     }
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
               
               // Get the physical node of this thread
               unsigned node = thread->runningOn()->getNUMANode();
               // Convert to virtual
               unsigned vNode = sys.getVirtualNUMANode( node );
               
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
               int deepTasksN = tdata._readyQueues[ nodeToQueue( vNode, false ) ].size();
               bool emptyBigTasks = tdata._readyQueues[ nodeToQueue( vNode, true )].empty();
               //fprintf( stderr, "[sockets] %d tasks at the small's queue\n", deepTasksN );
               //fprintf( stderr, "[sockets] %u tasks at the big's queue\n", (unsigned)tdata._readyQueues[ nodeToQueue( vNode, true )].size() );
               
               //unsigned thId = thread->getId();
               
               
               // TODO Improve atomic condition
               bool stealFromBig = deepTasksN < 1*sys.getCoresPerSocket() && !emptyBigTasks;
                   /*&& ( tdata._activeMasters[socket].value() == 0 || tdata._activeMasters[socket].value() == thId )*/
               unsigned queueNumber = nodeToQueue( vNode, stealFromBig );
               
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
                  // Index of the queue where we stole the WD
                  unsigned index;
                  
                  if ( false /* steal from the biggest */ )
                  {
                     // Find the queue with the most small tasks
                     WDPriorityQueue<> *largest = &tdata._readyQueues[2];
                     //int largestSocket = 0;
                     for ( int i = 1; i < sys.getNumSockets(); ++i )
                     {
                        WDPriorityQueue<> *current = &tdata._readyQueues[ (i+1)*2 ];
                        if ( largest->size() < current->size() ){
                           largest = current;
                           //largestSocket = i;
                           index = (i+1)*2;
                        }
                     }
                     //fprintf( stderr, "Stealing from socket #%d that has %lu tasks\n", largestSocket, largest->size() );
                     /*if( _stealLowPriority )
                        wd = largest->pop_back( thread );
                     else
                        wd = largest->pop_front( thread );*/
                  }
                  else if ( _randomSteal )
                  {
                     unsigned random = std::rand() % sys.getNumAvailSockets();
                     //index = random * 2 + offset;
                     index = nodeToQueue( random, _stealParents );
                  }
                  // Round robbin steal
                  else {
                     // getStealNext returns a physical node, we must convert it
                     int close = _nearSockets[node].getStealNext();
                     int vClose = sys.getVirtualNUMANode( close );
                     
                     // 2 queues per socket + 1 master queue + 1 (offset of the inner tasks)
                     index = nodeToQueue( vClose, _stealParents );
                  }
                  
                  if( _stealLowPriority )
                     wd = tdata._readyQueues[index].pop_back( thread );
                  else
                     wd = tdata._readyQueues[index].pop_front( thread );
                  
                  if ( wd != NULL ){
                     // Change queue
                     wd->setWakeUpQueue( index );
                     return wd;
                  }
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
                  if ( prefetchThread != NULL ) {
                     prefetchThread->addNextWD( &wd );
                     
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
            
            /*! \brief Enables or disables stealing */
            virtual void setStealing( bool value )
            {
               _steal = value;
            }
            
            /*! \brief Returns the status of stealing */
            virtual bool getStealing()
            {
               return _steal;
            }
      };

      class SocketSchedPlugin : public Plugin
      {
         private:
            int _numSockets;
            int _coresPerSocket;
            
            bool _steal;
            bool _stealParents;
            bool _stealLowPriority;
            bool _immediate;
            bool _smart;
            
            unsigned _spins;
            
            bool _random;
            
            void loadDefaultValues()
            {
               _numSockets = sys.getNumSockets();
               _coresPerSocket = sys.getCoresPerSocket();
            }
         public:
            SocketSchedPlugin() : Plugin( "Socket-aware scheduling Plugin",1 ),
               _steal( true ), _stealParents( false ), _stealLowPriority( false),
               _immediate( false ), _smart( false ),
               _spins( 200 ), _random( false ) {}

            virtual void config( Config& cfg ) {
               // Read hwloc's info before reading user parameters
               loadDefaultValues();
               
               cfg.setOptionsSection( "Sockets module", "Socket-aware scheduling module" );
               cfg.registerConfigOption( "socket-steal", NEW Config::FlagOption( _steal ), "Enable work stealing from other sockets' inner tasks queues (default)." );
               cfg.registerArgOption( "socket-steal", "socket-steal" );
               
               cfg.registerConfigOption( "socket-steal-parents", NEW Config::FlagOption( _stealParents ), "Steal top level tasks, instead of child tasks (disabled by default)." );
               cfg.registerArgOption( "socket-steal-parents", "socket-steal-parents" );
               
               cfg.registerConfigOption( "socket-steal-low-priority", NEW Config::FlagOption( _stealLowPriority ), "Steal low priority tasks, instead of higher priority tasks (disabled by default)." );
               cfg.registerArgOption( "socket-steal-low-priority", "socket-steal-low-priority" );
               
               cfg.registerConfigOption( "socket-immediate", NEW Config::FlagOption( _immediate ), "Use the immediate successor when prefecting (disabled by default)." );
               cfg.registerArgOption( "socket-immediate", "socket-immediate" );
               
               cfg.registerConfigOption( "socket-smartpriority", NEW Config::FlagOption( _smart ), "Propagates priority to the immediate predecessors (disabled by default)." );
               cfg.registerArgOption( "socket-smartpriority", "socket-smartpriority" );
               
               cfg.registerConfigOption( "socket-steal-spin", NEW Config::UintVar( _spins ), "Number of spins before stealing (200 by default)." );
               cfg.registerArgOption( "socket-steal-spin", "socket-steal-spin" );
               
               
               cfg.registerConfigOption( "socket-random-steal", NEW Config::FlagOption( _random ), "Steal from random sockets instead of round robin (disabled by default)." );
               cfg.registerArgOption( "socket-random-steal", "socket-random-steal" );
            }

            virtual void init() {
               //fprintf(stderr, "Setting numSockets to %d and coresPerSocket to %d\n", _numSockets, _coresPerSocket );
               sys.setNumSockets( _numSockets );
               sys.setCoresPerSocket( _coresPerSocket );
               
               sys.setDefaultSchedulePolicy( NEW SocketSchedPolicy( _steal, _stealParents, _stealLowPriority, _immediate, _smart, _spins, _random ) );
            }
      };

   }
}

DECLARE_PLUGIN("sched-socket",nanos::ext::SocketSchedPlugin);
