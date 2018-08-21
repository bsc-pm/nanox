/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#include "system.hpp"
#include "network_decl.hpp"
#include "remoteworkdescriptor_decl.hpp"

using namespace nanos;

RemoteWorkDescriptor::RemoteWorkDescriptor() : WorkDescriptor( 0, NULL ), _destinationNode ( (unsigned int) -1 ) {
}

RemoteWorkDescriptor::RemoteWorkDescriptor( unsigned int destinationNode ) : WorkDescriptor( 0, NULL ), _destinationNode ( destinationNode ) {
   if ( sys.getPMInterface().getInternalDataSize() > 0 ) {
      char *data = NEW char[sys.getPMInterface().getInternalDataSize()];
      sys.getPMInterface().initInternalData( data );
      this->setInternalData( data );
   }
}

void RemoteWorkDescriptor::exitWork( WorkDescriptor &work ) { 
   WorkDescriptor::exitWork( work );
   sys.getNetwork()->sendWorkDoneMsg( _destinationNode, work.getRemoteAddr() );
}
