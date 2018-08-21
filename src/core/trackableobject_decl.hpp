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

#ifndef _NANOS_TRACKABLE_OBJECT_DECL_H
#define _NANOS_TRACKABLE_OBJECT_DECL_H
#include <stdlib.h>
#include <list>
#include "dependableobject_decl.hpp"
#include "commutationdepobj_decl.hpp"
#include "atomic_decl.hpp"
#include "lock_decl.hpp"

namespace nanos {

  /*! \class TrackableObject
   *  \brief Object associated to an address with which different DependableObject operate
   */
   class TrackableObject
   {
      public:
         typedef std::list< DependableObject *> DependableObjectList; /**< Type list of DependableObject */
      private:
         DependableObject      *_lastWriter; /**< Points to the last DependableObject registered as writer of the TrackableObject */
         DependableObjectList   _versionReaders; /**< List of readers of the last version of the object */
         Lock                   _readersLock; /**< Lock to provide exclusive access to the readers list */
         Lock                   _writerLock; /**< Lock internally the object for secure access to _lastWriter */
         CommutationDO         *_commDO; /**< Will be successor of all commutation tasks using this object untill a new reader/writer appears */
         bool                   _hold; /**< Cannot be erased since it is in use */
      public:

        /*! \brief TrackableObject default constructor
         *
         *  Creates a TrackableObject with the given address associated.
         */
         TrackableObject ()
            : _lastWriter ( NULL ), _versionReaders(), _readersLock(), _writerLock(), _commDO(NULL), _hold(false) {}

        /*! \brief TrackableObject copy constructor
         *
         *  \param obj another TrackableObject
         */
         TrackableObject ( const TrackableObject &obj ) 
            :   _lastWriter ( obj._lastWriter ), _versionReaders(), _readersLock(), _writerLock(), _commDO(NULL), _hold(false) {}

        /*! \brief TrackableObject destructor
         */
         ~TrackableObject () {}

        /*! \brief TrackableObject assignment operator, can be self-assigned.
         *
         *  \param obj another TrackableObject
         */
         const TrackableObject & operator= ( const TrackableObject &obj );

        /*! \brief Returns true if the TrackableObject has a DependableObject as LastWriter
         */
         bool hasLastWriter ( );

        /*! \brief Get the last writer
         *  \sa DependableObject
         */
         DependableObject* getLastWriter ( ) const;

        /*! \brief Set the last writer
         *  \sa DependableObject
         */
         void setLastWriter ( DependableObject &depObj );

        /*! \brief Delete the last writer if it matches the given one
         *  \param depObj DependableObject to compare with _lastWriter
         *  \sa DependableObject
         */
         void deleteLastWriter ( DependableObject &depObj );

        /*! \brief Get the list of readers
         *  \sa DependableObjectList
         */
         DependableObjectList const & getReaders ( ) const;

        /*! \brief Get the list of readers
         *  \sa DependableObjectList
         */
         DependableObjectList & getReaders ( );

        /*! \brief Add a new reader
         *  \sa DependableObject
         */
         void setReader ( DependableObject &reader );

        /*! \brief Returns true if do is reader of the TrackableObject
         */
         bool hasReader ( DependableObject &depObj );

        /*! \brief Delete all readers from the object
         */ 
         void flushReaders ( );

        /*! \brief Deletes a reader from the object's list
         */
         void deleteReader ( DependableObject &reader );

        /*! \brief Whether the object has readers or not
         */
         bool hasReaders ();

        /*! \brief Returns the readers lock
         */
         Lock& getReadersLock();

        /*! \brief Returns the commutationDO if it exists
         */
         CommutationDO* getCommDO() const;

        /*! \brief Set the commutationDO
         *  \param commDO to set in this object
         */
         void setCommDO( CommutationDO *commDO );
         
        /*! \brief Return whether the region is held and thus cannot be removed
         */
         bool isOnHold ( ) const;
         
        /*! \brief Holds the region so that it cannot be removed (by the same thread)
         */
         void hold ( );
         
        /*! \brief Unholds the region so that it can be removed (by the same thread)
         */
         void unhold ( );
         
        /*! \brief Returns if the region has no information
         */
         bool isEmpty ( );
   };

   //! \brief RegionStatus stream formatter
   //! \param o the output stream
   //! \param regionStatus the region status
   //! \returns the output stream
   inline std::ostream & operator<<( std::ostream &o, TrackableObject const &status);

} // namespace nanos

#endif
