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

#ifndef _NANOS_CLUSTER_NODE
#define _NANOS_CLUSTER_NODE

#include "config.hpp"
#include "simpleallocator_decl.hpp"
#include "clusterinfo_decl.hpp"
#include "clusterdevice_decl.hpp"
#include "smpdd.hpp"
#ifdef GPU_DEV
//FIXME: GPU support
#include "gpudd.hpp"
#endif
#include "cachedaccelerator.hpp"

namespace nanos {
   namespace ext {

      class ClusterNode : public CachedAccelerator
      {

         private:
            // config variables
            static Atomic<int>      _deviceSeed; // Number of cluster devices assigned to threads
            unsigned int            _clusterNode; // Assigned cluster device Id

            // disable copy constructor and assignment operator
            ClusterNode( const ClusterNode &pe );
            const ClusterNode & operator= ( const ClusterNode &pe );

            SimpleAllocator _memSegment;
            unsigned int _executedWorkDesciptors;

         public:
            // constructors
#ifdef GPU_DEV
            ClusterNode( int id ) : CachedAccelerator( id, &SMP, &GPU, &Cluster, ClusterInfo::getSegmentLen( id ) ),
#else
            ClusterNode( int id ) : CachedAccelerator( id, &SMP, NULL, &Cluster, ClusterInfo::getSegmentLen( id ) ),
#endif
            _clusterNode ( id ), _memSegment( ( uintptr_t ) ClusterInfo::getSegmentAddr( id ),
                  ClusterInfo::getSegmentLen( id ) ), _executedWorkDesciptors ( 0 ) { }

            virtual ~ClusterNode() {}

            virtual WD & getWorkerWD () const;
            virtual WD & getMasterWD () const;
            virtual WD & getMultiWorkerWD () const
            {
               fatal( "getMultiWorkerWD: ClusterNode is not allowed to create MultiThreads" );
            }
            virtual BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );
            virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs )
            {
               fatal( "ClusterNode is not allowed to create MultiThreads" );
            }

            // capability query functions
            virtual bool supportsUserLevelThreads () const { return false; }
            virtual bool isGPU () const { return false; }
            virtual bool supportsDirectTransfersWith( ProcessingElement const &pe ) const;
            virtual unsigned int getMyNodeNumber() const { return _clusterNode; }

            unsigned int getClusterNodeNum() const;
            SimpleAllocator & getAllocator( void ) { return _memSegment; }

            void incExecutedWDs() { _executedWorkDesciptors++; }
            unsigned int getExecutedWDs() { return _executedWorkDesciptors; }
            unsigned int getNodeNum() { return _clusterNode; }
      };
   }
}

#endif
