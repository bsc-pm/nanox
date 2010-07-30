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

#include "clustermsg.hpp"
#include "clusterremotenode.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;
using namespace nanos::ext;

//void ClusterRemoteNode::stopAll()
//{
//    fprintf(stderr, "I have to send a message to a remote node.\n");
//    ClusterMsg::sendFinishMessage(this);
//}
//Atomic<int> ClusterRemoteNode::_deviceSeed = 0;

void ClusterRemoteNode::slaveLoop ( ClusterRemoteNode *node)
{
   Scheduler::workerLoop();
   sys.getNetwork()->sendExitMsg( node->getClusterNodeNum() );
}

WorkDescriptor & ClusterRemoteNode::getWorkerWD () const
{
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct ) slaveLoop );
   WD *wd = new WD( dd, sizeof(this), (void *) this );
   return *wd;
}

WorkDescriptor & ClusterRemoteNode::getMasterWD () const
{
   fatal("Attempting to create a cluster master thread");
}

#if 0
BaseThread &ClusterRemoteNode::createThread ( WorkDescriptor &helper )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   ClusterThread &th = *new ClusterThread( helper,this, _clusterDevice );

   return th;
}

void ClusterRemoteNode::registerDataAccessDependent( uint64_t tag, size_t size )
{
   //_cache.cacheData( tag, size );
}

void ClusterRemoteNode::copyDataDependent( uint64_t tag, size_t size )
{
   //_cache.copyData( tag, size );
}

void ClusterRemoteNode::unregisterDataAccessDependent( uint64_t tag )
{
   //_cache.flush( tag );
}

void ClusterRemoteNode::copyBackDependent( uint64_t tag, size_t size )
{
   //_cache.copyBack( tag, size );
}

void* ClusterRemoteNode::getAddressDependent( uint64_t tag )
{
   //return _cache.getAddress(tag);
   return 0;
}

void ClusterRemoteNode::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   //_cache.copyTo( dst, tag, size );
}

void ClusterRemoteNode::registerCacheAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterRemoteNode::unregisterCacheAccessDependent(uint64_t a, size_t aa){}
void ClusterRemoteNode::registerPrivateAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterRemoteNode::unregisterPrivateAccessDependent(uint64_t a, size_t aa){}
#endif

