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
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <limits>

#ifdef HWLOC
#include <hwloc.h>
#endif

#ifdef GPU_DEV
#include "gpudd.hpp"
#endif

#include "memcachecopy.hpp"
#include "globalregt.hpp"


namespace nanos {
   namespace ext {
      
      struct SocketSchedConfig
      {
         //! \brief Limit stealing to adjacent nodes (1 hop away)
         bool stealFromAdjacent;
         
         SocketSchedConfig() : stealFromAdjacent( true ) {}
      };

      class SocketSchedPolicy : public SchedulePolicy
      {
         public:
            const static int UnassignedNode =-1; //!< Value returned by getNode() when it does not find a suitable nod

            using SchedulePolicy::queue;         //!< Load default implementation of queue prior to redefinition
            using SchedulePolicy::successorFound;//!< Load default implementation of successorFound
         
         private:
            bool _steal;                         //!< Steal work from other sockets?
            bool _stealParents;                  //!< Steal top level tasks (true) or children (false)
            bool _stealLowPriority;              //!< Steal low priority tasks (true) or high (false)
            bool _useSuccessor;                  //!< Use immediate successor when prefecting
            bool _smartPriority;                 //!< Use smart priority (propagate priority)
            unsigned _spins;                     //!< Number of loops to spin before attempting stealing
            bool _randomSteal;                   //!< If enabled, steal from random queues, otherwise round robin
            bool _useCopies;                     //!< Uses copy information to detect the NUMA nodes

            SocketSchedConfig _config;           //!< Configuration for the policy
            
            typedef std::vector<unsigned> NearSocketsList; //!< For a given socket, a list of near sockets.
            
            //! \brief Info related to socket distance and stealing 
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
            
            typedef std::vector<SocketDistanceInfo> NearSocketsMatrix; //!< For each socket, an array of close sockets
            NearSocketsMatrix _nearSockets;      //!< Keep a vector with all the close sockets for every socket
            std::set<int> _gpuNodes;             //!< A set of all the nodes with GPUs
            SocketDistanceInfo _gpuNodesToGive;  //!< List of gpu nodes that will be used to give away work descriptors in round robin

            struct TeamData : public ScheduleTeamData
            {
               WDPriorityQueue<>*         _readyQueues;
               Atomic<unsigned>           _next; //!< Next queue to insert to (round robin scheduling) TODO remove this since we don't use it
               Atomic<bool>*              _activeMasters; //!< If there is an active "master" thread, for every socket
 
               TeamData ( unsigned int sockets ) : ScheduleTeamData(), _next( 0 )
               {
                  _readyQueues = NEW WDPriorityQueue<>[ sockets*2 + 1 ];
                  _activeMasters = NEW Atomic<bool>[ sockets ];
               }

               ~TeamData () {
                  delete[] _readyQueues;
                  delete[] _activeMasters;
               }
            };

            //! \brief Socket Scheduler data associated to each thread
            struct ThreadData : public ScheduleThreadData
            {
               unsigned int _cacheId;
               bool _init;

               ThreadData () : _cacheId(0), _init(false) {}
               virtual ~ThreadData () {}
            };
            
            struct WDData : public ScheduleWDData
            {
               bool _initTask;
               unsigned int _wakeUpQueue;
               
               WDData () : _initTask( false ), _wakeUpQueue( std::numeric_limits<unsigned>::max() ) {}
               virtual ~WDData() {}
            };
         
            //! \brief Comparison functor used in distance computations
            struct DistanceCmp {
               unsigned* _distances;
               
               DistanceCmp( unsigned* distances ) : _distances( distances ) {}
               
               bool operator()( unsigned socket1, unsigned socket2 )
               {
                  return _distances[socket1] < _distances[socket2];
               }
            };

            // disable copy and assigment
            explicit SocketSchedPolicy ( const SocketSchedPolicy & );
            const SocketSchedPolicy & operator= ( const SocketSchedPolicy & );
         
         private:
            //! \brief Creates lists of close sockets.
            void computeDistanceInfo() {
               // Fill the distance info matrix
               _nearSockets.resize( sys.getSMPPlugin()->getNumSockets() );
               
               // For every numa node
               for ( unsigned from = 0; from < _nearSockets.size(); ++from )
               {
                  _nearSockets[ from ].stealNext = 0;
                  NearSocketsList& row = _nearSockets[ from ].list;
                  row.reserve( sys.getSMPPlugin()->getNumSockets() - 1 );
                  
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
                  
                  for ( unsigned to = 0; to < (unsigned) sys.getSMPPlugin()->getNumSockets(); ++to ) {
                     if ( to == from || sys.getVirtualNUMANode( to ) == INT_MIN ) continue;
                     row.push_back( to );
                  }
                  if ( !row.empty() ) {
                     // Sort by distance
                     std::sort( row.begin(), row.end(), DistanceCmp( distances ) );
                     // Keep only close nodes
                     if ( _config.stealFromAdjacent ) {
                        NearSocketsList::iterator it = std::upper_bound(
                           row.begin(), row.end(), row.front(),
                           DistanceCmp( distances )
                        );
                        row.erase( it, row.end() );
                     }
                  }
               }
            }
            
            //! \brief Finds all the nodes with GPUs. Constructs a set of nodes that do have GPUs, so it can be queried later on.
            void findGPUNodes()
            {
               // Look for GPUs in all the nodes
               for (System::ThreadList::iterator it=sys.getWorkersBegin();
                  it!=sys.getWorkersEnd(); it++) {
#ifdef GPU_DEV
                  BaseThread *worker = it->second;
                  if ( worker->runningOn()->supports( nanos::ext::GPU ) )
                  {
                     int node = worker->runningOn()->getNumaNode();
                     // Convert to virtual
                     int vNode = sys.getVirtualNUMANode( node );
                     _gpuNodes.insert( vNode );
                     verbose0( "[NUMA] Found GPU Worker in node " << node << " (virtual " << vNode << ")" );
                  }
#endif
               }
               // Initialise the structure that will cycle through gpu nodes when giving away GPU tasks
               _gpuNodesToGive.stealNext = 0;
               // Copy the set elements to a list (since it is easier to cycle through)
               std::copy( _gpuNodes.begin(), _gpuNodes.end(),
                  std::back_inserter( _gpuNodesToGive.list ) );
            }
            
            //! \brief Checks if a wd can be run in a node.
            //! If a GPU task is being scheduled in a node with no GPU threads, it will return false.
            inline bool canRunInNode( WD& wd, int node )
            {
#ifdef GPU_DEV
               // If it's a GPU wd and the node has GPUs, it can run
               if ( wd.canRunIn( nanos::ext::GPU ) ) {
                  //fprintf( stderr, "GPU WD, can it run in node %d? %d\n", node, (int)_gpuNodes.count( node ) );
                  return _gpuNodes.count( node ) != 0;
               }
#endif
               // FIXME: Assume it's SMP and can always run in this node. This might not be always true.
               return true;
            }
            
            inline int findBetterNode( WD& wd, int node )
            {
               // We do not use round robbin here, we just send it to the closest node
               int newNode = _gpuNodesToGive.getStealNext();
               fatal_cond( _gpuNodes.count( newNode ) == 0, "Cannot find a node with GPUs to move the task to." );
               
               
               verbose( "[NUMA] WD " << wd.getId() << " cannot run in node " << node << ", moved to " << newNode );
               return newNode;
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
            
            /*!
             *  \brief Checks if the WD is an initialisation task.
             */
            inline bool isInitTask( const WD& wd ) const
            {
               if ( wd.getNumCopies() == 0 )
                  return 0;
               
               const CopyData * copies = wd.getCopies();
               //unsigned int wo_copies = 0, ro_copies = 0, rw_copies = 0;
               std::size_t createdDataSize = 0;
               for ( unsigned int idx = 0; idx < wd.getNumCopies(); ++idx )
               {
                  if ( !copies[idx].isPrivate() ) {
                     if ( wd._mcontrol._memCacheCopies[ idx ].getVersion() == 1 && copies[idx].isOutput() )
                        createdDataSize += copies[idx].getSize();
                  }
               }
               
               //return wo_copies + ro_copies == wd.getNumCopies();
               return createdDataSize > 0;
            }
            
            /*!
             *  \brief Returns the node this WD should run on, based on copies
             *  information if _useCopies is enabled.
             *  Otherwise, it will use the node set by the user.
             *
             *  It will also set the WD NUMA node when using copies, since in
             *  that case that property is set to -1.
             *
             *  Init tasks will be distributed in round robbin.
             *  FIXME (gmiranda): round robin should be for available nodes!
             *
             *  If there's a tie (i.e. some nodes have as much data as others),
             *  the node will be chosen randomly.
             *
             * \retval UnassignedNode If no suitable node was found to execute
             * this task, meaning that it should go to the general queue.
             * \retval >=0 The node to execute this task that has at least as
             * much data as the others.
             */
            inline int getNode( BaseThread *thread, WD& wd ) const
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               WDData & wdata = *dynamic_cast<WDData*>( wd.getSchedulerData() );

               // If copies are disabled, simply return the node set by current_socket
               if ( !_useCopies )
                  return wd.getNUMANode();

               const CopyData * copies = wd.getCopies();
               unsigned numNodes = sys.getNumNumaNodes();
               
               int winner;
               
               if( isInitTask( wd ) )
               {
                  wdata._initTask = true;
                  winner = tdata._next++ % sys.getNumNumaNodes();
                  
                  verbose0( toString( "[NUMA] wd ") + toString( wd.getId() ) + toString( "(" ) + toString( wd.getDescription() )
                     + toString(")") + toString( " is init task, assigned to NUMA node " ) + toString( winner ) );
                  //fprintf( stderr, "[socket] Round.robbin next = %d\n", tdata._next.value() );
               }
               else
               {
                  unsigned int numaRanks[ numNodes ];
                  std::fill( numaRanks, numaRanks + numNodes, 0 );
                  
                  for ( unsigned int i = 0; i < wd.getNumCopies(); i++ ) {
                     if ( !copies[i].isPrivate() && ( copies[i].isInput() || copies[i].isOutput() ) ) {
                     //if ( !copies[i].isPrivate() && copies[i].isInput() && copies[i].isOutput() ) {
                        NewLocationInfoList const &locs = wd._mcontrol._memCacheCopies[ i ]._locations;
                        if ( locs.empty() ) {
                           //std::cerr << "empty list, version "<<  wd._mcontrol._memCacheCopies[ i ]._version << std::endl;
                           const ProcessingElement * loc = wd._mcontrol._memCacheCopies[ i ]._reg.getFirstWriterPE();
                           if ( loc != NULL )
                           {
                              int numaNode = loc->getNumaNode();
                              numaRanks[ numaNode ] += wd._mcontrol._memCacheCopies[ i ]._reg.getDataSize();
                           }
                        } else {
                           for ( NewLocationInfoList::const_iterator it = locs.begin(); it != locs.end(); it++ ) {
                              global_reg_t reg( it->first, wd._mcontrol._memCacheCopies[ i ]._reg.key );
                              
                              const ProcessingElement * loc = reg.getFirstWriterPE();
                              
                              if ( loc != NULL  )
                              {
                                 int pNumaNode = loc->getNumaNode();
                                 int vNumaNode = sys.getVirtualNUMANode(pNumaNode);
                                 numaRanks[ vNumaNode ] += reg.getDataSize();
                              }
                           }
                        }
                     }
                  }
                  
                  #if 0
                  fprintf(stderr, "[NUMA] Numa ranks for wd %d (%s): {", wd.getId(), wd.getDescription() );
                  for( unsigned int x = 0; x < sys.getNumNumaNodes(); ++x )
                  {
                     fprintf(stderr, "%d, ", numaRanks[ x ] );
                  }
                  fprintf(stderr, "}\n" );
                  
                  #endif
                  
                  winner = UnassignedNode;
                  // Nodes with the highest rank (equal!)
                  std::set<unsigned> candidateRanks;
                  // Highest rank until now.
                  unsigned int maxRank = 0;
                  // FIXME: review the use of start
                  unsigned int start = 0 ;
                  // Find the nodes with the highest rank
                  for ( unsigned i = start; i < ( sys.getNumNumaNodes() ); i++ ) {
                     // If this rank is higher than the previous one,
                     if ( numaRanks[i] > maxRank ) {
                        // Discard all previous nodes
                        candidateRanks.clear();
                        // Add this one to the set
                        candidateRanks.insert( i );
                        // And save the new max rank
                        maxRank = numaRanks[i];
                        continue;
                     }
                     /* If this rank is the same as the previous max
                        and max != 0 to prevent having ranks with 0 bytes in
                        the set */
                     if ( numaRanks[i] == maxRank && maxRank != 0 ) {
                        // Simply add this node to the candidate set
                        candidateRanks.insert( i );
                        continue;
                     }
                  }
                  /* Now we've got a set with nodes that are equally good.
                   * Always choosing the same would not be wise: there would be imbalance,
                   * that node will always get more tasks.
                   * Round robbin would work if all the tasks access the same nodes,
                   * but if task A is accessed by nodes #0 and #1, you give it to node #0,
                   * the next task should be given to node #1. Then task B comes and it is
                   * accessed by nodes #2 and #3, so you can't give it to node #1.
                   * I think random is the best way for now.
                   */
                  // If there's a tie
                  if ( candidateRanks.size() > 1 ) {
                     unsigned pos = std::rand() % candidateRanks.size();
                     std::set<unsigned>::const_iterator it( candidateRanks.begin() );
                     // Move the iterator to the random position
                     advance( it, pos );
                     winner = *it;
                     verbose0( toString( "[NUMA] Tie resolved, candidate is pos: " ) + toString( pos ) + toString( " (node " ) + toString( winner ) );
                  }
                  // If there's only one element
                  else if ( candidateRanks.size() == 1 ) {
                     winner = *( candidateRanks.begin() );
                  }
                  // Otherwise, it seems there's no NUMA access
                  else {
                     winner = UnassignedNode;
                  }
               }

               verbose0( "[NUMA] Winner is " + toString( winner ) );

               wd.setNUMANode( winner );

               return winner;
            }

         public:
            // constructor
            SocketSchedPolicy ( bool steal, bool stealParents, bool stealLowPriority,
               bool useSuccessor, bool smartPriority,
               unsigned spins, bool randomSteal, bool useCopies,
               SocketSchedConfig& config  )
               : SchedulePolicy ( "Socket" ), _steal( steal ),
               _stealParents( stealParents ), _stealLowPriority( stealLowPriority ),
               _useSuccessor( useSuccessor ), _smartPriority( smartPriority ),
               _spins ( spins ), _randomSteal( randomSteal ), _useCopies( useCopies ),
               _config( config )
            {}

            // destructor
            virtual ~SocketSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               if ( _steal && sys.getNumNumaNodes() == 1 )
               {
                  message0( "[NUMA] Stealing can not be enabled with just one NUMA node available, disabling it" );
                  _steal = false;
               }

               // Now we can find GPU nodes since the workers will have been created
               findGPUNodes();

               computeDistanceInfo();

               // Create 2 queues per socket plus one for the global queue.
               return NEW TeamData( sys.getNumNumaNodes() );
            }

            virtual std::string getSummary() const
            {
               std::ostringstream s;
               s << "====================== NUMA Summary ======================" << std::endl;
               s << "=== Worker binding:" << std::endl;
               for (System::ThreadList::iterator it=sys.getWorkersBegin();
                     it!=sys.getWorkersEnd(); it++) {
                  const BaseThread *w = it->second;

                  s << "===  | Worker " << w->getId() << ", cpu id: " << w->getCpuId()
                     << ", NUMA node " << w->runningOn()->getNumaNode() << std::endl;
               }

               std::stringstream ss;
               std::ostream_iterator<int> outIt (ss ,", ");
               std::copy( _gpuNodes.begin(), _gpuNodes.end(), outIt );
               s << "=== CUDA devices in:     " << ss.str() << std::endl;

               //// Clear stringstream to reuse it
               //ss.str( std::string() );

               s << "=== NUMA node mapping (virtual -> physical):   ";
               const std::vector<int> &numaNodeMap = sys.getNumaNodeMap();
               for ( int pNode = 0; pNode < (int)numaNodeMap.size(); ++pNode )
               {
                  int vNode = numaNodeMap[pNode];
                  // If this real node has a valid virtual mapping
                  if ( vNode != INT_MIN )
                     s << vNode << "->" << pNode << ", ";
               }
               s << std::endl;
               return s.str();
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
            }
            
            virtual size_t getWDDataSize () const { return sizeof( WDData ); }
            virtual size_t getWDDataAlignment () const { return __alignof__( WDData ); }
            virtual void initWDData ( void * data ) const
            {
               NEW (data)WDData();
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
               WDData & wdata = *dynamic_cast<WDData*>( wd.getSchedulerData() );
               unsigned index = wdata._wakeUpQueue;
               // FIXME: use another variable to check this condition
               // If the WD has not been distributed yet, distribute it
               if ( index == std::numeric_limits<unsigned>::max() )
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
                     if ( wakeUp ) tdata._readyQueues[index].push_front ( &wd );
                     else tdata._readyQueues[index].push_back ( &wd );
                     break;
                  default:
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
               WDData & wdata = *dynamic_cast<WDData*>( wd.getSchedulerData() );
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               
               if( wdata._wakeUpQueue != std::numeric_limits<unsigned>::max() )
                  warning0( "WD already has a queue (" << wdata._wakeUpQueue << ")" );
               
               unsigned index;
               unsigned node;
               
               switch( wd.getDepth() ) {
                  case 0:
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  case 1:
                     node = ( unsigned ) getNode( thread, wd );
                     // If a node was not selected (either by the user, or because there were no copies)
                     if ( wd.getNUMANode() == UnassignedNode )
                        // Go to the general queue
                        index = 0;
                     // Otherwise, do the usual stuff.
                     else
                     {
                        // Use copy information if enabled, otherwise, use info by nanos_current_socket()
                        // If the node cannot execute this WD
                        if ( !canRunInNode( wd, node ) ){
                           node = findBetterNode( wd, node );
                        }
                        
                        // 2 queues per socket, the first one is for level 1 tasks
                        fatal_cond( node >= sys.getNumNumaNodes(), "Invalid node selected" );
                        index = nodeToQueue( node, true );
                        wdata._wakeUpQueue = index;
                     }
                     
                     // Insert at the front (these will have higher priority)
                     tdata._readyQueues[index].push_back ( &wd );
                     break;
                  default:
                     // Insert this in its parent's node
                     index = wdata._wakeUpQueue;
                     
                     node = queueToNode( index );
                     // If this wd cannot run in this node
                     if ( !canRunInNode( wd, node ) ) {
                        node = findBetterNode( wd, node );
                        fatal_cond( node >= sys.getNumNumaNodes(), "Invalid node selected" );
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
                     wdata._wakeUpQueue = index;
                     
                     // Insert at the back
                     tdata._readyQueues[index].push_back ( &wd );
                     break;
               }
            }

            //! \brief Function called when a new task must be created.
            //! \param thread pointer to the thread to which belongs the new task
            //! \param wd a reference to the work descriptor of the new task
            //! \sa WD and BaseThread
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               distribute( thread, newWD );

               return 0;
            }

            virtual WD * atIdle ( BaseThread *thread, int numSteal )
            {
               WorkDescriptor * wd = thread->getNextWD();
               if ( wd ) return wd;

               // If stealing has been enabled and its time to steal
               if ( numSteal && _steal )
                  // Try...
                  return stealWork( thread );
               
               // Get the physical node of this thread
               unsigned node = thread->runningOn()->getNumaNode();
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
               
               // TODO Improve atomic condition
               // Note (gmiranda): For true nested operation
               bool parentQueue = deepTasksN < 1*sys.getSMPPlugin()->getCPUsPerSocket() && !emptyBigTasks;
               
               unsigned queueNumber = nodeToQueue( vNode, parentQueue );
               
               wd = tdata._readyQueues[queueNumber].pop_front( thread );
               
               if ( wd != NULL ) return wd;
               
               // If this queue is empty, try the global queue
               return tdata._readyQueues[0].pop_front( thread );
            }
            
            WD * stealWork ( BaseThread *thread )
            {
               unsigned index; //!< Index of the queue where we stole the WD
               WD* wd = NULL;  //!< WD to return
               
               // Get schedule data
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();

               // Get the physical node of this thread
               unsigned node = thread->runningOn()->getNumaNode();
               
               
               if ( _randomSteal ) {
                  unsigned random = std::rand() % sys.getNumNumaNodes();
                  // index = random * 2 + offset;
                  index = nodeToQueue( random, _stealParents );
               } else {
                  // getStealNext returns a physical node, we must convert it
                  int close = _nearSockets[node].getStealNext();
                  int vClose = sys.getVirtualNUMANode( close );
                  
                  // 2 queues per socket + 1 master queue + 1 (offset of the inner tasks)
                  index = nodeToQueue( vClose, _stealParents );
               }
               
               if ( _stealLowPriority ) wd = tdata._readyQueues[index].pop_back( thread );
               else wd = tdata._readyQueues[index].pop_front( thread );
               
               if ( wd != NULL ) {
                  WDData & wdata = *dynamic_cast<WDData*>( wd->getSchedulerData() );
                  // Change queue
                  wdata._wakeUpQueue = index;
                  return wd;
               }
               
               return NULL;
            }
            
            virtual WD * atWakeUp( BaseThread *thread, WD &wd )
            {
               // If the WD was waiting for something
               if ( wd.started() ) {
                  BaseThread * prefetchThread = NULL;
                  // Check constraints since they won't be checked in Schedule::wakeUp
                  if ( Scheduler::checkBasicConstraints ( wd, *thread ) ) {
                     prefetchThread = thread;
                  } else {
                     prefetchThread = wd.isTiedTo();
                  }
                  
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
               if ( !_useSuccessor ) {
                  // Revert to the base behaviour
                  return SchedulePolicy::atPrefetch( thread, current );
               }
               
               WD * found = current.getImmediateSuccessor(*thread);
            
               return found != NULL ? found : atIdle(thread,false);
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
            void successorFound( DependableObject &successor, DependableObject *predecessor, int mode, int numPred )
            {
               if ( !_smartPriority || mode )
                  return;
               
               debug( "SmartPriority::successorFound" );
               if ( predecessor == NULL ) {
                  debug( "SmartPriority::successorFound predecessor is NULL" );
                  return;
               }
               
               WD *pred = ( WD* ) predecessor->getRelatedObject();
               if ( pred == NULL ) {
                  debug( "SmartPriority::successorFound predecessor->getRelatedObject() is NULL" )
                  return;
               }
               
               WD *succ = ( WD* ) successor.getRelatedObject();
               if ( succ == NULL ) {
                  fatal( "SmartPriority::successorFound  successor->getRelatedObject() is NULL" );
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
                  WDData & wdata = *dynamic_cast<WDData*>( pred->getSchedulerData() );
                  unsigned index = wdata._wakeUpQueue;

                  // What happens if pred is not in any queue? Fatal.
                  if ( index < static_cast<unsigned>( sys.getSMPPlugin()->getNumSockets() ) ) {
                     tdata._readyQueues[ index ].reorderWD( pred );
                  }
               }
            }

            //! \brief Enables or disables stealing
            virtual void setStealing( bool value ) { _steal = value; }
            
            //! \brief Returns the status of stealing
            virtual bool getStealing() { return _steal; }
            
            //! \brief Returns if scheduler uses priorities 
            bool usingPriorities() const { return true; }

            bool testDequeue()
            {
               TeamData &tdata = (TeamData &) *myThread->getTeam()->getScheduleData();
               int num_queues = sys.getSMPPlugin()->getNumSockets()*2 + 1;
               for ( int i=0; i<num_queues; ++i ) {
                  if ( tdata._readyQueues[i].testDequeue() ) return true;
               }
               return false;
            }
      };

      class SocketSchedPlugin : public Plugin
      {
         private:
            
            bool _steal;
            bool _stealParents;
            bool _stealLowPriority;
            bool _immediate;
            bool _smart;
            
            unsigned _spins;
            
            bool _random;
            
            bool _useCopies;
            
            SocketSchedConfig _schedConfig;
            void loadDefaultValues() { }
         public:
            SocketSchedPlugin() : Plugin( "Socket-aware scheduling Plugin",1 ),
               _steal( true ), _stealParents( false ), _stealLowPriority( false),
               _immediate( false ), _smart( false ),
               _spins( 200 ), _random( false ), _useCopies( false ) {}

            virtual void config( Config& cfg ) {
               
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
               
               cfg.registerConfigOption( "socket-auto-detect", NEW Config::FlagOption( _useCopies ), "Automatic NUMA node assignment based on copy information and detection of initialisation tasks (disabled by default)." );
               cfg.registerArgOption( "socket-auto-detect", "socket-auto-detect" );

               cfg.registerConfigOption( "socket-steal-adjacent", NEW Config::FlagOption( _schedConfig.stealFromAdjacent ), "Limit stealing to adjacent nodes (default)");
               cfg.registerArgOption( "socket-steal-adjacent", "socket-steal-adjacent" );
            }

            virtual void init() {
               // Read hwloc's info before reading user parameters
               loadDefaultValues();
               
               sys.setDefaultSchedulePolicy( NEW SocketSchedPolicy( _steal, _stealParents, _stealLowPriority, _immediate, _smart, _spins, _random, _useCopies, _schedConfig ) );
            }
      };

   }
}

DECLARE_PLUGIN("sched-socket",nanos::ext::SocketSchedPlugin);
