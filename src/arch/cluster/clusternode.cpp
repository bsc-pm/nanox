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
#include "clusternode.hpp"
#include "clusterdd.hpp"
#include "clusterthread.hpp"
#include "clusterdevice.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;
using namespace nanos::ext;

//void ClusterNode::slaveLoop ( ClusterNode *node)
//{
//   Scheduler::workerLoop();
//   sys.getNetwork()->sendExitMsg( node->getClusterNodeNum() );
//}

WorkDescriptor & ClusterNode::getWorkerWD () const
{
   ClusterDD * dd = new ClusterDD( ( ClusterDD::work_fct )Scheduler::workerClusterLoop );
   WD *wd = new WD( dd );
   std::cerr << "c:node @ is " << (void * ) this << " id " << _clusterNode << " wd is " << wd << ":" << wd->getId() << std::endl;
   wd->setPe( (ProcessingElement *) this );
   wd->unsetClusterMigrable();
   return *wd;
}

WorkDescriptor & ClusterNode::getMasterWD () const
{
   fatal("Attempting to create a cluster master thread");
}

BaseThread &ClusterNode::createThread ( WorkDescriptor &helper )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   ClusterThread &th = *new ClusterThread( helper, this, _clusterNode );

   return th;
}

//void ClusterNode::registerDataAccessDependent( uint64_t tag, size_t size )
//{
//   _cache.cacheData( tag, size );
//}

//void ClusterNode::copyDataDependent( uint64_t tag, size_t size )
//{
//   _cache.copyData( tag, size );
//}

//void ClusterNode::unregisterDataAccessDependent( uint64_t tag )
//{
//   _cache.flush( tag );
//}

//void ClusterNode::copyBackDependent( uint64_t tag, size_t size )
//{
//   _cache.copyBack( tag, size );
//}

void* ClusterNode::getAddressDependent( uint64_t tag )
{
   return _cache.getAddress(tag);
}

void ClusterNode::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   _cache.copyTo( dst, tag, size );
}

void ClusterNode::registerCacheAccessDependent( uint64_t tag, size_t size, bool input, bool output )
{
   //fprintf(stderr, "clusternode: registerCacheAccessDependent %llx\n", tag);
   _cache.registerCacheAccess( tag, size, input, output );
}

void ClusterNode::unregisterCacheAccessDependent( uint64_t tag, size_t size, bool output )
{
   _cache.unregisterCacheAccess( tag, size, output );
}

void ClusterNode::registerPrivateAccessDependent( uint64_t tag, size_t size, bool input, bool output )
{
   _cache.registerPrivateAccess( tag, size, input, output );
}

void ClusterNode::unregisterPrivateAccessDependent( uint64_t tag, size_t size )
{
   _cache.unregisterPrivateAccess( tag, size );
}

unsigned int ClusterNode::getClusterNodeNum() { return _clusterNode; }

void ClusterNode::waitInputDependent( uint64_t tag ) {}
