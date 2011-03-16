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
#include "simpleallocator.hpp"
#include "clusterinfo.hpp"
#include "clusterdevice.hpp"
#include "clusterdd.hpp"
//FIXME: GPU support
//#include "gpudd.hpp"
#include "cachedaccelerator.hpp"

namespace nanos {
namespace ext {

   class ClusterNode : public CachedAccelerator< ClusterDevice, WriteBackPolicy >
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
         //FIXME: GPU support
         ClusterNode( int id ) : CachedAccelerator< ClusterDevice, WriteBackPolicy >( sys.getNumPEs(), &Cluster, NULL/*&GPU*/, ( int ) ClusterInfo::getSegmentLen( id ) ), 
            _clusterNode ( id ), _memSegment( ( uintptr_t ) ClusterInfo::getSegmentAddr( id ), ClusterInfo::getSegmentLen( id ) ), _executedWorkDesciptors ( 0 ) { }

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

         unsigned int getClusterNodeNum();
         SimpleAllocator & getAllocator( void ) { return _memSegment; }

         void incExecutedWDs() { _executedWorkDesciptors++; }
         unsigned int getExecutedWDs() { return _executedWorkDesciptors; }
         unsigned int getNodeNum() { return _clusterNode; }
   };
}
}

#endif
