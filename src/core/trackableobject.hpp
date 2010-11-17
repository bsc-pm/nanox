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

#ifndef _NANOS_TRACKABLE_OBJECT
#define _NANOS_TRACKABLE_OBJECT
#include <stdlib.h>
#include <list>
#include <algorithm>
#include "dependableobject.hpp"
#include "atomic.hpp"

namespace nanos
{

  /*! \class TrackableObject
   *  \brief Object associated to an address with which different DependableObject operate
   */
   class TrackableObject
   {
      public:
         typedef std::list< DependableObject *> DependableObjectList; /**< Type list of DependableObject */
      private:
         void                  * _address; /**< Pointer to the dependency address */
         DependableObject      *_lastWriter; /**< Points to the last DependableObject registered as writer of the TrackableObject */
         DependableObjectList   _versionReaders; /**< List of readers of the last version of the object */
         Lock                   _readersLock; /**< Lock to provide exclusive access to the readers list */
         Lock                   _writerLock; /**< Lock internally the object for secure access to _lastWriter */
      public:
        /*! \brief TrackableObject default constructor
         *
         *  Creates a TrackableObject with the given address associated.
         */
         TrackableObject ( void * address = NULL )
            : _address(address), _lastWriter ( NULL ), _versionReaders(), _readersLock(), _writerLock() {}
        /*! \brief TrackableObject copy constructor
         *
         *  \param obj another TrackableObject
         */
         TrackableObject ( const TrackableObject &obj ) 
            :  _address ( obj._address ), _lastWriter ( obj._lastWriter ), _versionReaders(), _readersLock(), _writerLock() {}
        /*! \brief TrackableObject destructor
         */
         ~TrackableObject () {}
        /*! \brief TrackableObject assignment operator, can be self-assigned.
         *
         *  \param obj another TrackableObject
         */
         const TrackableObject & operator= ( const TrackableObject &obj )
         {
            _address = obj._address;
            _lastWriter = obj._lastWriter;
            return *this;
         }
        /*! \brief Obtain the address associated to the TrackableObject
         */
         void * getAddress ( )
         {
            return _address;
         }
        /*! \brief Returns true if the TrackableObject has a DependableObject as LastWriter
         */
         bool hasLastWriter ( )
         {
            return _lastWriter != NULL;
         }
        /*! \brief Get the last writer
         *  \sa DependableObject
         */
         DependableObject* getLastWriter ( )
         {
            return _lastWriter;
         }
        /*! \brief Set the last writer
         *  \sa DependableObject
         */
         void setLastWriter ( DependableObject &depObj )
         {
            _writerLock.acquire();
            memoryFence();

            _lastWriter = &depObj;

            memoryFence();
            _writerLock.release();
         }
        /*! \brief Delete the last writer if it matches the given one
         *  \param depObj DependableObject to compare with _lastWriter
         *  \sa DependableObject
         */
         void deleteLastWriter ( DependableObject &depObj )
         {
            if ( _lastWriter == &depObj ) {

               _writerLock.acquire();
               memoryFence();

               if ( _lastWriter ==  &depObj ) {
                  _lastWriter = NULL;
               }

               memoryFence();
               _writerLock.release();
            }
         }
        /*! \brief Get the list of readers
         *  \sa DependableObjectList
         */
         DependableObjectList & getReaders ( )
         {
            return _versionReaders;
         }
        /*! \brief Add a new reader
         *  \sa DependableObject
         */
         void setReader ( DependableObject &reader )
         {
            _versionReaders.push_back( &reader );
         }
        /*! \brief Returns true if do is reader of the TrackableObject
         */
         bool hasReader ( DependableObject &depObj )
         {
            return ( find( _versionReaders.begin(), _versionReaders.end(), &depObj ) != _versionReaders.end() );
         }
        /*! \brief Delete all readers from the object
         */ 
         void flushReaders ( )
         {
            _versionReaders.clear();
         }
        /*! \brief Deletes a reader from the object's list
         */
         void deleteReader ( DependableObject &reader )
         {
            _versionReaders.remove( &reader );
         }
        /*! \brief Whether the object has readers or not
         */
         bool hasReaders ()
         {
            return !( _versionReaders.empty() );
         }
        /*! \brief Get exclusive access to the readers list
         */
         void lockReaders ( )
         {
            _readersLock.acquire();
            memoryFence();
         }
        /*! \brief Release the readers' list lock
         */
         void unlockReaders ( )
         {
            memoryFence();
            _readersLock.release();
         }
   };

};

#endif
