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
#ifdef HWLOC
#include <hwloc.h>
#endif

namespace nanos {
   namespace ext {

      class SocketSchedPolicy : public SchedulePolicy
      {
         public:
         
         private:

            struct TeamData : public ScheduleTeamData
            {
               WDDeque*           _readyQueues;
               //! Next queue to insert to (round robin scheduling)
               Atomic<unsigned>           _next;
 
               TeamData ( unsigned int size ) : ScheduleTeamData(), _next( 0 )
               {
                  _readyQueues = NEW WDDeque[size];
               }

               ~TeamData () { delete[] _readyQueues; }
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

            /* disable copy and assigment */
            explicit SocketSchedPolicy ( const SocketSchedPolicy & );
            const SocketSchedPolicy & operator= ( const SocketSchedPolicy & );

         public:
            // constructor
            SocketSchedPolicy() : SchedulePolicy ( "Socket" )
            {
               int numSockets = sys.getNumSockets();
               int coresPerSocket = sys.getCoresPerSocket();

               // Check config
               if ( numSockets != std::ceil( sys.getNumPEs() / static_cast<float>( coresPerSocket) ) )
               {
                  unsigned validSockets = std::ceil( sys.getNumPEs() / static_cast<float>(coresPerSocket) );
                  warning0( "Adjusting num-sockets from " << numSockets << " to " << validSockets );
                  sys.setNumSockets( validSockets );
               }
            }

            // destructor
            virtual ~SocketSchedPolicy() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return sizeof(ThreadData); }

            virtual ScheduleTeamData * createTeamData ()
            {
               // Create as many queues as sockets we have plus one for the
               // global queue.
               return NEW TeamData( sys.getNumSockets() + 1 );
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return NEW ThreadData();
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
            virtual void queue ( BaseThread *thread, WD &wd )
            {
               TeamData &tdata = (TeamData &) *thread->getTeam()->getScheduleData();
               
               unsigned index;
               
               switch( wd.getDepth() ) {
                  case 0:
                     //fprintf( stderr, "Wake up Depth 0, inserting WD %d in queue number 0\n", wd.getId() );
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  // Keep other tasks in the same socket as they were
                  // Note: we might want to insert the ones with depth 1 in the front
                  default:
                     index = wd.getWakeUpQueue();
                     
                     //fprintf( stderr, "Wake up Depth >0, inserting WD %d in queue number %d\n", wd.getId(), index );
                     
                     // Insert at the front (these will have higher priority)
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
               if( wd.getWakeUpQueue() != 0 )
                  fprintf( stderr, "Queue: warning: wd already has a queue (%d)\n", wd.getWakeUpQueue());
               
               unsigned index;
               
               switch( wd.getDepth() ) {
                  case 0:
                     //fprintf( stderr, "Depth 0, inserting WD %d in queue number 0\n", wd.getId() );
                     // Implicit WDs, insert them in the general queue.
                     tdata._readyQueues[0].push_back ( &wd );
                     break;
                  case 1:
                     index = (tdata._next++ ) % sys.getNumSockets() + 1;
                     wd.setWakeUpQueue( index );
                     
                     //fprintf( stderr, "Depth 1, inserting WD %d in queue number %d\n", wd.getId(), index );
                     
                     // Insert at the front (these will have higher priority)
                     tdata._readyQueues[index].push_front ( &wd );
                     
                     // Round robin
                     //tdata._next = ( socket+1 ) % sys.getNumSockets();
                     //fprintf( stderr, "Next = %d\n", tdata._next.value() );
                     break;
                  default:
                     // Insert this in its parent's socket
                     index = wd.getParent()->getWakeUpQueue();
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
               
               wd = tdata._readyQueues[socket+1].pop_front( thread );
               
               if ( wd != NULL )
                  return wd;
               
               // If this queue is empty, try the global queue
               return tdata._readyQueues[0].pop_front( thread );
            }
            
            virtual WD * atWakeUp( BaseThread *thread, WD &wd )
            {
               queue( thread, wd );
               
               return NULL;
            }
            
            /*virtual WD * atBlock( BaseThread *thread, WD *current )
            {
               return atWakeUp( thread, *current );
            }*/
      };

      class SocketSchedPlugin : public Plugin
      {
         private:
            int _numSockets;
            int _coresPerSocket;
            
            void loadDefaultValues()
            {
               // Read number of sockets and cores from hwloc
#ifdef HWLOC
               hwloc_topology_t topology;
               
               /* Allocate and initialize topology object. */
               hwloc_topology_init( &topology );
               
               /* Perform the topology detection. */
               hwloc_topology_load( topology );
               int depth = hwloc_get_type_depth( topology, HWLOC_OBJ_SOCKET );
               
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
               fprintf(stderr, "No hwloc support\n" );
               _numSockets = sys.getNumSockets();
               _coresPerSocket = sys.getCoresPerSocket();
#endif
            }
         public:
            SocketSchedPlugin() : Plugin( "Socket-aware scheduling Plugin",1 ) {}

            virtual void config( Config& cfg ) {
               // Read hwloc's info before reading user parameters
               loadDefaultValues();
               
               cfg.setOptionsSection( "Sockets module", "Socket-aware scheduling module" );
   
               cfg.registerConfigOption( "cores-per-socket", NEW Config::PositiveVar( _coresPerSocket ), "Number of cores per socket." );
               cfg.registerArgOption( "cores-per-socket", "cores-per-socket" );
               
               cfg.registerConfigOption( "num-sockets", NEW Config::PositiveVar( _numSockets ), "Number of sockets available." );
               cfg.registerArgOption( "num-sockets", "num-sockets" );
            }

            virtual void init() {
               fprintf(stderr, "Setting numSockets to %d and coresPerSocket to %d\n", _numSockets, _coresPerSocket );
               sys.setNumSockets( _numSockets );
               sys.setCoresPerSocket( _coresPerSocket );
               
               sys.setDefaultSchedulePolicy(NEW SocketSchedPolicy());
            }
      };

   }
}

DECLARE_PLUGIN("sched-socket",nanos::ext::SocketSchedPlugin);
