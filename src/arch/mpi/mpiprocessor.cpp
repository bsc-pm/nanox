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

#include "mpiprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "mpithread.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

bool MPIProcessor::_useUserThreads = true;
size_t MPIProcessor::_threadsStackSize = 0;
System::CachePolicyType MPIProcessor::_cachePolicy = System::DEFAULT;
size_t MPIProcessor::_cacheDefaultSize = 1048580;

MPIProcessor::MPIProcessor( int id , MPI_Comm communicator, int rank ) : CachedAccelerator<MPIDevice>( id, &MPI ) 
{
             _communicator=communicator;
             _rank=rank;
}


void MPIProcessor::prepareConfig ( Config &config )
{
   config.registerConfigOption( "user-threads", NEW Config::FlagOption( _useUserThreads, false), "Disable use of user threads to implement workdescriptor" );
   config.registerArgOption( "user-threads", "disable-ut" );

   config.registerConfigOption ( "pthreads-stack-size", NEW Config::SizeVar( _threadsStackSize ), "Defines pthreads stack size" );
   config.registerArgOption( "pthreads-stack-size", "pthreads-stack-size" );

}

WorkDescriptor & MPIProcessor::getWorkerWD () const
{
   MPIDD * dd = NEW MPIDD( ( MPIDD::work_fct )Scheduler::workerLoop );
   WD *wd = NEW WD( dd );
   return *wd;
}

WorkDescriptor & MPIProcessor::getMasterWD () const
{
   WD * wd = NEW WD( NEW MPIDD() );
   return *wd;
}

BaseThread &MPIProcessor::createThread ( WorkDescriptor &helper )
{
   ensure( helper.canRunIn( SMP ),"Incompatible worker thread" );
   MPIThread &th = *NEW MPIThread( helper,this);
   th.stackSize( _threadsStackSize ).useUserThreads( _useUserThreads );

   return th;
}

