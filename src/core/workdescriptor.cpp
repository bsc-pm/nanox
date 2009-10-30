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

#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include "debug.hpp"

using namespace nanos;

DeviceData * WorkDescriptor::findDeviceData ( const Device &device ) const
{
   for ( int i = 0; i < num_devices; i++ ) {
      if ( devices[i]->isCompatible( device ) ) {
         return devices[i];
      }
   }

   return 0;
}

DeviceData & WorkDescriptor::activateDevice ( const Device &device )
{
   if ( active_device ) {
      ensure( active_device->isCompatible( device ),"Bogus double device activation" );
      return *active_device;
   }

   DD * dd = findDeviceData( device );

   ensure( dd,"Did not find requested device in activation" );
   active_device = dd;
   return *dd;
}

bool WorkDescriptor::canRunIn( const Device &device ) const
{
   if ( active_device ) return active_device->isCompatible( device );

   return findDeviceData( device ) != NULL;
}

bool WorkDescriptor::canRunIn ( const ProcessingElement &pe ) const
{
   return canRunIn( pe.getDeviceType() );
}
