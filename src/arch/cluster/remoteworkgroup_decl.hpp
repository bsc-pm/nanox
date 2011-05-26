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

#ifndef _NANOS_REMOTE_WORK_GROUP_DECL_H
#define _NANOS_REMOTE_WORK_GROUP_DECL_H

#include "system_decl.hpp"
#include "workgroup_decl.hpp"
namespace nanos
{
   class RemoteWorkGroup : public WorkGroup
   {
      private:
         unsigned int _remoteId;
      public:
         RemoteWorkGroup(unsigned int rId) : _remoteId ( rId ) {}
      
         virtual void exitWork( WorkGroup &work ) { 
		sys.getNetwork()->sendWorkDoneMsg( 
			Network::MASTER_NODE_NUM, 
			/*new queue */work.getRemoteAddr() , _remoteId
                        /* (void *) work.getId() */ 
                        /*(void *) _remoteId*/ ); }
   };

};


#endif

