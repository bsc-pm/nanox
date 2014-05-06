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

#ifndef _NANOS_COMMUTATIONDEPOBJ
#define _NANOS_COMMUTATIONDEPOBJ
#include "commutationdepobj_decl.hpp"
#include "task_reduction.hpp"

using namespace nanos;

inline void CommutationDO::dependenciesSatisfied ( )
{
   DependenciesDomain *domain = getDependenciesDomain( );
   if ( domain ) {
      domain->removeCommDO ( this, *_target );
   }
   if ( _taskReduction != NULL ) {
      void *addr = _taskReduction->finalize();
#ifndef ON_TASK_REDUCTION
      myThread->getTeam()->removeTaskReduction( addr );
#else
      //FIXME no only current WD
      myThread->getCurrentWD()->removeTaskReduction( addr );
#endif

   }
   
   finished();
}

inline bool CommutationDO::isCommutative() const 
{ 
   return _commutative; 
} 

inline void CommutationDO::setTaskReduction( TaskReduction *tr )
{
   _taskReduction = tr;
}

#endif


