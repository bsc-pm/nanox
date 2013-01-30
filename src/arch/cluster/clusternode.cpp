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

#include <iostream>
#include "clusternode_decl.hpp"
#include "clusterthread_decl.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "cachedaccelerator.hpp"

using namespace nanos;
using namespace nanos::ext;


ClusterNode::ClusterNode( int id ) : CachedAccelerator( id, &SMP, 
#ifdef GPU_DEV
   &GPU,
#else
   NULL,
#endif
   &Cluster, ClusterInfo::getSegmentLen( id ), RegionCache::ALLOC_WIDE ), _clusterNode ( id ),
   _memSegment( ( uintptr_t ) ClusterInfo::getSegmentAddr( id ),
   ClusterInfo::getSegmentLen( id ) ), _executedWorkDesciptors ( 0 ) {
}

ClusterNode::~ClusterNode() {
}

WorkDescriptor & ClusterNode::getWorkerWD () const {
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct )0xdeadbeef );
   WD *wd = new WD( dd );
   return *wd;
}

WorkDescriptor & ClusterNode::getMasterWD () const {
   fatal("Attempting to create a cluster master thread");
}

WD & ClusterNode::getMultiWorkerWD () const {
   fatal( "getMultiWorkerWD: ClusterNode is not allowed to create MultiThreads" );
}

BaseThread &ClusterNode::createThread ( WorkDescriptor &helper, SMPMultiThread *parent ) {
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   ClusterThread &th = *new ClusterThread( helper, this, parent, _clusterNode );

   return th;
}

BaseThread & ClusterNode::createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs ) {
   fatal( "ClusterNode is not allowed to create MultiThreads" );
}

bool ClusterNode::supportsUserLevelThreads () const {
   return false;
}

bool ClusterNode::isGPU () const {
   return false;
}

bool ClusterNode::supportsDirectTransfersWith( ProcessingElement const &pe ) const {
   return ( &Cluster == pe.getCacheDeviceType() && sys.useNode2Node() );
}

unsigned int ClusterNode::getMyNodeNumber() const {
   return _clusterNode;
}

unsigned int ClusterNode::getClusterNodeNum() const {
   return _clusterNode;
}

SimpleAllocator & ClusterNode::getAllocator( void ) {
   return _memSegment;
}

void ClusterNode::incExecutedWDs() {
   _executedWorkDesciptors++;
}

unsigned int ClusterNode::getExecutedWDs() const {
   return _executedWorkDesciptors;
}

unsigned int ClusterNode::getNodeNum() const {
   return _clusterNode;
}
