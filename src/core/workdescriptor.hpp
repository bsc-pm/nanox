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
#include "dependableobjectwd.hpp"


namespace nanos
{

// forward declarations

    class BaseThread;

    class ProcessingElement;

    class WDDeque;

    class Device
    {

        private:
            const char *_name;

        public:
            // constructor
            Device ( const char *n ) : _name ( n ) {}

            // copy constructor
            Device ( const Device &arch ) : _name ( arch._name ) {}

            // assignment operator
            const Device & operator= ( const Device &arch )
            {
                _name = arch._name;
                return *this;
            }

            // destructor
            ~Device() {};

            bool operator== ( const Device &arch ) {
                return arch._name == _name;
            }
    };

// This class holds the specific data for a given device

    class DeviceData
    {

        private:
            // use pointers for this as is this fastest way to compare architecture
            // compatibility
            const Device *_architecture;

        public:
            // constructors
            DeviceData ( const Device *arch ) : _architecture ( arch ) {}

            // copy constructor
            DeviceData ( const DeviceData &dd ) : _architecture ( dd._architecture )  {}

            // assignment operator
            const DeviceData & operator= ( const DeviceData &dd )
            {
                  // self-assignment: ok
                  _architecture = dd._architecture;
                  return *this;
            }

            bool isCompatible ( const Device &arch )
            {
                return _architecture == &arch;
            }

            // destructor
            virtual ~DeviceData() {}
    };

    class WorkDescriptor : public WorkGroup
    {

        private:
            void    *            _data;
            void    *            _wdData; // this allows higher layer to associate data to the WD
            bool                 _tie;
            BaseThread *         _tiedTo;
            bool                 _idle;

            //Added parent for cilk scheduler: first steal parent task, next other tasks
            WorkDescriptor *     _parent;

            //Added reference to queue to allow dequeuing from third party (e.g. cilk scheduler)
            WDDeque *            _myQueue;

            //level (depth) of the task
            unsigned              _depth;

            // Supported devices for this workdescriptor
            unsigned             _numDevices;
            DeviceData **        _devices;
            DeviceData *         _activeDevice;

            DOSubmit _dOSubmit;
            DOWait _dOWait;

            DependenciesDomain _depsDomain;

            WorkDescriptor ( const WorkDescriptor &wd );
            const WorkDescriptor & operator= ( const WorkDescriptor &wd );

        public:
            // constructors
            WorkDescriptor ( int ndevices, DeviceData **devs,void *wdata=0 ) :
                    WorkGroup(), _data ( wdata ), _wdData ( 0 ), _tie ( false ), _tiedTo ( 0 ), _idle ( false ),
                    _parent ( NULL ), _myQueue ( NULL ), _depth ( 0 ), _numDevices ( ndevices ), _devices ( devs ), _activeDevice ( ndevices == 1 ? devs[0] : 0 ), _dOSubmit(this), _dOWait(this), _depsDomain() {}

            WorkDescriptor ( DeviceData *device,void *wdata=0 ) :
                    WorkGroup(), _data ( wdata ), _wdData ( 0 ), _tie ( false ), _tiedTo ( 0 ), _idle ( false ),
                    _parent ( NULL ), _myQueue ( NULL ), _depth ( 0 ), _numDevices ( 1 ), _devices ( &_activeDevice ),
                    _activeDevice ( device ), _dOSubmit(this), _dOWait(this), _depsDomain() {}

            // destructor
            // all data will be allocated in a single chunk so only the destructors need to be invoked
            // but not the allocator
            virtual ~WorkDescriptor()
            {
               for ( unsigned i = 0; i < _numDevices; i++ )
                  _devices[i]->~DeviceData();
            }

            WorkDescriptor * getParent() {
                return _parent;
            }

            void setParent ( WorkDescriptor * p ) {
                _parent = p;
            }

            WDDeque * getMyQueue() {
                return _myQueue;
            }

            void setMyQueue ( WDDeque * myQ ) {
                _myQueue = myQ;
            }

            bool isEnqueued() {
                return ( _myQueue != NULL );
            }

            /* named arguments idiom */
            WorkDescriptor & tied () {
                _tie = true; return *this;
            }

            WorkDescriptor & tieTo ( BaseThread &pe ) {
                _tiedTo = &pe; _tie=false; return *this;
            }

            bool isTied() const {
                return _tiedTo != NULL;
            }

            BaseThread * isTiedTo() const {
                return _tiedTo;
            }

            void setData ( void *wdata ) {
                _data = wdata;
            }

            void * getData () const {
                return _data;
            }

            bool isIdle () const {
                return _idle;
            }

            void setIdle ( bool state=true ) {
                _idle = state;
            }

            void setDepth ( int l ) {
                _depth = l;
            }

            unsigned getDepth() {
                return _depth;
            }

            /* device related methods */
            DeviceData * findDeviceData ( const Device &device ) const;
            bool canRunIn ( const Device &device ) const;
            bool canRunIn ( const ProcessingElement &pe ) const;
            DeviceData & activateDevice ( const Device &device );
            DeviceData & getActiveDevice () const {
                return *_activeDevice;
            }

            bool hasActiveDevice() const {
                return _activeDevice != NULL;
            }

            void setInternalData ( void *data ) {
                _wdData = data;
            }

            void * getInternalData () const {
                return _wdData;
            }

            void submitWithDependencies( WorkDescriptor &wd, int numDeps, Dependency* deps )
            {
               _depsDomain.submitDependableObject( wd._dOSubmit, numDeps, deps );
            }

            void waitOn( int numDeps, Dependency* deps )
            {
               _depsDomain.submitDependableObject( _dOWait, numDeps, deps );
            }
   
            void workFinished(WorkDescriptor &wd)
            {
               _depsDomain.finished( wd._dOSubmit );
            }

    };

    typedef class WorkDescriptor WD;

    typedef class DeviceData DD;

};

#endif

