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

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

bool SMPProcessor::_useUserThreads = true;
size_t SMPProcessor::_threadsStackSize = 0;

void SMPProcessor::prepareConfig ( Config &config )
{
   config.registerConfigOption( "user-threads", new Config::FlagOption( _useUserThreads, false), "Disable use of user threads to implement workdescriptor" );
   config.registerArgOption( "user-threads", "disable-ut" );

   config.registerConfigOption ( "pthreads-stack-size", new Config::SizeVar( _threadsStackSize ), "Defines pthreads stack size" );
   config.registerArgOption( "pthreads-stack-size", "pthreads-stack-size" );
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
   SMPDD * dd = new SMPDD( ( SMPDD::work_fct )Scheduler::idle );
   WD *wd = new WD( dd );
   return *wd;
}

WorkDescriptor & SMPProcessor::getMasterWD () const
{
   WD * wd = new WD( new SMPDD() );
   return *wd;
}

BaseThread &SMPProcessor::createThread ( WorkDescriptor &helper )
{
   ensure( helper.canRunIn( SMP ),"Incompatible worker thread" );
   SMPThread &th = *new SMPThread( helper,this );
   th.stackSize(_threadsStackSize).useUserThreads(_useUserThreads);

   return th;
}


#if SMP_NUMA

void SMPProcessor::registerDataAccessDependent( uint64_t tag, size_t size )
{
   _cache.cacheData( tag, size );
}

void SMPProcessor::copyDataDependent( uint64_t tag, size_t size )
{
   _cache.copyData( tag, size );
}

void SMPProcessor::unregisterDataAccessDependent( uint64_t tag )
{
   _cache.flush( tag );
}

void SMPProcessor::copyBackDependent( uint64_t tag, size_t size )
{
   _cache.copyBack( tag, size );
}

void* SMPProcessor::getAddressDependent( uint64_t tag )
{
   return _cache.getAddress(tag);
}

void SMPProcessor::copyToDependent( void *dst, uint64_t tag, size_t size )
{
   _cache.copyTo( dst, tag, size );
}

#endif

