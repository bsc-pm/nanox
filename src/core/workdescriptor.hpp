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

#ifndef _NANOS_WORK_DESCRIPTOR
#define _NANOS_WORK_DESCRIPTOR

#include <stdlib.h>
#include <utility>
#include <vector>
#include "workgroup.hpp"


namespace nanos
{

// forward declarations

   class BaseThread;

   class ProcessingElement;

   class WDDeque;

   class Device
   {

      private:
         const char *name;

      public:
         // constructor
         Device( const char *n ) : name( n ) {}

         // copy constructor
         Device( const Device &arch ) : name( arch.name ) {}

         // assignment operator
         const Device & operator= ( const Device &arch ) { name = arch.name; return *this; }

         // destructor
         ~Device() {};

         bool operator== ( const Device &arch ) { return arch.name == name; }
   };

// This class holds the specific data for a given device

   class DeviceData
   {

      private:
         // use pointers for this as is this fastest way to compare architecture
         // compatibility
         const Device *architecture;

      public:
         // constructors
         DeviceData( const Device *arch ) : architecture( arch ) {}

         // copy constructor
         DeviceData( const DeviceData &dd ) : architecture( dd.architecture )  {}

         // assignment operator
         const DeviceData & operator= ( const DeviceData &wd );

         bool isCompatible( const Device &arch ) { return architecture == &arch; }

         // destructor
         virtual ~DeviceData() {}
   };

   class WorkDescriptor : public WorkGroup
   {

      private:
         void    *data;
         void    *wd_data; // this allows higher layer to associate data to the WD
         bool	  tie;
         BaseThread *  tie_to;
         bool      idle;

         //Added parent for cilk scheduler: first steal parent task, next other tasks
         WorkDescriptor * parent;

         //Added reference to queue to allow dequeuing from third party (e.g. cilk scheduler)
         WDDeque * myQueue;

         //level (depth) of the task
         int level;

         // Supported devices for this workdescriptor
         int         num_devices;
         DeviceData **devices;
         DeviceData *active_device;

      public:
         // constructors
         WorkDescriptor( int ndevices, DeviceData **devs,void *wdata=0 ) :
               WorkGroup(), data( wdata ), wd_data( 0 ), tie( false ), tie_to( 0 ), idle( false ),
               parent( NULL ), myQueue( NULL ), level( 0 ), num_devices( ndevices ), devices( devs ), active_device( 0 ) {}

         WorkDescriptor( DeviceData *device,void *wdata=0 ) :
               WorkGroup(), data( wdata ), wd_data( 0 ), tie( false ), tie_to( 0 ), idle( false ),
               parent( NULL ), myQueue( NULL ), level( 0 ), num_devices( 1 ), devices( 0 ), active_device( device ) {}

         // TODO: copy constructor
         WorkDescriptor( const WorkDescriptor &wd );
         // TODO: assignment operator
         const WorkDescriptor & operator= ( const WorkDescriptor &wd );
         // destructor
         virtual ~WorkDescriptor() { /*TODO*/ }

         WorkDescriptor * getParent() { return parent;}

         void setParent( WorkDescriptor * p ) {parent = p;}

         WDDeque * getMyQueue() {return myQueue;}

         void setMyQueue( WDDeque * myQ ) {myQueue = myQ;}

         bool isEnqueued() {return ( myQueue != NULL );}

         /* named arguments idiom */
         WorkDescriptor & tied () { tie = true; return *this; }

         WorkDescriptor & tieTo ( BaseThread &pe ) { tie_to = &pe; tie=false; return *this; }

         bool isTied() const { return tie_to != NULL; }

         BaseThread * isTiedTo() const { return tie_to; }

         void setData ( void *wdata ) { data = wdata; }

         void * getData () const { return data; }

         bool isIdle () const { return idle; }

         void setIdle( bool state=true ) { idle = state; }

         void setLevel( int l ) { level = l; }

         int getLevel() { return level; }

         /* device related methods */
         DeviceData * findDeviceData ( const Device &device ) const;
         bool canRunIn ( const Device &device ) const;
         bool canRunIn ( const ProcessingElement &pe ) const;
         DeviceData & activateDevice ( const Device &device );
         DeviceData & getActiveDevice () const { return *active_device; }

         bool hasActiveDevice() const { return active_device != NULL; }

         void setInternalData ( void *data ) { wd_data = data; }

         void * getInternalData () const { return wd_data; }

   };

   inline const DeviceData & DeviceData::operator= ( const DeviceData &dd )
   {
      // self-assignment: ok
      architecture = dd.architecture;
      return *this;
   }

   typedef class WorkDescriptor WD;

   typedef class DeviceData DD;

};

#endif

