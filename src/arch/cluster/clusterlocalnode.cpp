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

#include "clusterlocalnode.hpp"
#include "debug.hpp"
#include "schedule.hpp"

using namespace nanos;
using namespace nanos::ext;

void ClusterLocalNode::stopAll()
{
    fprintf(stderr, "I have to stop this node\n.");
}
#if 0
Atomic<int> ClusterProcessor::_deviceSeed = 0;
WorkDescriptor & ClusterProcessor::getWorkerWD () const
{
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   WD *wd = new WD( dd );
   return *wd;
}

WorkDescriptor & ClusterProcessor::getMasterWD () const
{
   fatal("Attempting to create a cluster master thread");
}

BaseThread &ClusterProcessor::createThread ( WorkDescriptor &helper )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   ClusterThread &th = *new ClusterThread( helper,this, _clusterDevice );

   return th;
}

void ClusterProcessor::registerDataAccessDependent( uint64_t tag, size_t size )
{
   //_cache.cacheData( tag, size );
}

void ClusterProcessor::copyDataDependent( uint64_t tag, size_t size )
{
   //_cache.copyData( tag, size );
}

void ClusterProcessor::unregisterDataAccessDependent( uint64_t tag )
{
   //_cache.flush( tag );
}

void ClusterProcessor::copyBackDependent( uint64_t tag, size_t size )
{
   //_cache.copyBack( tag, size );
}

void* ClusterProcessor::getAddressDependent( uint64_t tag )
{
   //return _cache.getAddress(tag);
   return 0;
}

void ClusterProcessor::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   //_cache.copyTo( dst, tag, size );
}

void ClusterProcessor::registerCacheAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterProcessor::unregisterCacheAccessDependent(uint64_t a, size_t aa){}
void ClusterProcessor::registerPrivateAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa){}
void ClusterProcessor::unregisterPrivateAccessDependent(uint64_t a, size_t aa){}
#endif


