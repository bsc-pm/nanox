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
