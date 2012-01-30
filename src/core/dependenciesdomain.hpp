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

#ifndef _NANOS_DEPENDENCIES_DOMAIN
#define _NANOS_DEPENDENCIES_DOMAIN
#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include "dependenciesdomain_decl.hpp"
#include "atomic.hpp"
#include "dependableobject.hpp"
#include "regiontree.hpp"
#include "regionstatus.hpp"
#include "dataaccess.hpp"
#include "compatibility.hpp"


using namespace nanos;

inline DependenciesDomain::DependenciesDomain ( ) :  _id( _atomicSeed++ ), _lastDepObjId ( 0 ), _regionMap( )
{
}

inline DependenciesDomain::DependenciesDomain ( const DependenciesDomain &depDomain )
   : _id( _atomicSeed++ ), _lastDepObjId ( depDomain._lastDepObjId ),
      _regionMap ( depDomain._regionMap )
{
}

inline DependenciesDomain::~DependenciesDomain ( )
{
}

inline int DependenciesDomain::getId()
{
   return _id;
}

inline void DependenciesDomain::submitDependableObject ( DependableObject &depObj, std::vector<DataAccess> const &dataAccesses )
{
   submitDependableObjectInternal ( depObj, dataAccesses.begin(), dataAccesses.end() );
}

inline void DependenciesDomain::submitDependableObject ( DependableObject& depObj, size_t numDataAccesses, const DataAccess* dataAccesses)
{
   submitDependableObjectInternal ( depObj, dataAccesses, dataAccesses+numDataAccesses );
}

inline RecursiveLock& DependenciesDomain::getInstanceLock()
{
   return _instanceLock;
}

inline Lock& DependenciesDomain::getLock()
{
   return _lock;
}

inline void DependenciesDomain::lock ( )
{
   _lock.acquire();
   memoryFence();
}

inline void DependenciesDomain::unlock ( )
{
   memoryFence();
   _lock.release();
}

#endif

