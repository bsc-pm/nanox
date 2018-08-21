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


#ifndef _NANOX_NETWORK
#define _NANOX_NETWORK

namespace nanos {

   class NetworkMessage
   {
      private:
      unsigned int _sequenceNumber;
      unsigned int _destination;
      public:
      NetworkMessage( dest ) : _sequenceNumber( seed++ ), _destination( dest );
      unsigned int getSequenceNumber() { return _sequenceNumber; }
   }

   class RemoteWorkDescriptorExecMsg : public NetworkMessage
   {
      private:
      public:
      RemoteWorkDescriptorExecMsg( const WD &wd ) : NetworkMessage( ) 
      {
      }
   }

} // namespace nanos

#endif
