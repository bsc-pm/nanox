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

#include "clusternode.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;
using namespace nanos::ext;

//Atomic<int> ClusterNode::_deviceSeed = 0;

WorkDescriptor & ClusterNode::getWorkerWD () const
{
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   WD *wd = new WD( dd );
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
   ClusterThread &th = *new ClusterThread( helper,this, _clusterDevice );

   return th;
}

void ClusterNode::registerDataAccessDependent( uint64_t tag, size_t size )
{
   //_cache.cacheData( tag, size );
}

void ClusterNode::copyDataDependent( uint64_t tag, size_t size )
{
   //_cache.copyData( tag, size );
}

void ClusterNode::unregisterDataAccessDependent( uint64_t tag )
{
   //_cache.flush( tag );
}

void ClusterNode::copyBackDependent( uint64_t tag, size_t size )
{
   //_cache.copyBack( tag, size );
}

void* ClusterNode::getAddressDependent( uint64_t tag )
{
   //return _cache.getAddress(tag);
   return 0;
}

void ClusterNode::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   //_cache.copyTo( dst, tag, size );
}

void ClusterNode::registerCacheAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterNode::unregisterCacheAccessDependent(uint64_t a, size_t aa){}
void ClusterNode::registerPrivateAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterNode::unregisterPrivateAccessDependent(uint64_t a, size_t aa){}


int ClusterNode::getClusterID() { return _clusterDevice; }
