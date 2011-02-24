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

WorkDescriptor & ClusterNode::getWorkerWD () const
{
   ClusterDD * dd = new ClusterDD( ( ClusterDD::work_fct )0xdeadbeef );
   WD *wd = new WD( dd );
   std::cerr << "c:node @ is " << (void * ) this << " id " << _clusterNode << " wd is " << wd << ":" << wd->getId() << std::endl;
   return *wd;
}

WorkDescriptor & ClusterNode::getMasterWD () const
{
   fatal("Attempting to create a cluster master thread");
}

BaseThread &ClusterNode::createThread ( WorkDescriptor &helper, SMPMultiThread *parent )
{
   // In fact, the GPUThread will run on the CPU, so make sure it canRunIn( SMP )
   ensure( helper.canRunIn( SMP ), "Incompatible worker thread" );
   ClusterThread &th = *new ClusterThread( helper, this, parent, _clusterNode );

   return th;
}

unsigned int ClusterNode::getClusterNodeNum() { return _clusterNode; }
