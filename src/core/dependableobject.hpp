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

#ifndef _NANOS_DEPENDABLE_OBJECT
#define _NANOS_DEPENDABLE_OBJECT
#include <stdlib.h>
#include <list>
#include <vector>
#include "atomic.hpp"
#include "trackableobject_fwd.hpp"

namespace nanos
{
  /*! \brief Abstract entity submitted to the Dependency system
   */
   class DependableObject
   {
      public:
         /**< Type vector of successors  */
         typedef std::vector<DependableObject *> DependableObjectVector;
         /**< Type vector of output objects */
         typedef std::vector<TrackableObject *> TrackableObjectVector;
         
      private:
         /**< DependableObject identifier */
         unsigned int _id;
         /**< Number of predecessors locking this object */
         Atomic<unsigned int> _numPredecessors;
         /** References counter */
         unsigned int _references;

         /**< List of successiors */
         DependableObjectVector _successors;

         /**< List of output objects */
         TrackableObjectVector _outputObjects;
         
         /**< List of read objects */
         TrackableObjectVector _readObjects;

         /**< Lock to do exclusive use of the DependableObject */
         Lock _objectLock;

      public:
        /*! \brief Constructor
         */
         DependableObject ( ) :  _id ( 0 ), _numPredecessors ( 0 ), _references(1), _successors(), _outputObjects(), _readObjects(), _objectLock() {}

        /*! \brief Copy constructor
         *  \param depObj another DependableObject
         */
         DependableObject ( const DependableObject &depObj ) :  _id ( depObj._id ), _numPredecessors ( depObj._numPredecessors ), _references(depObj._references), _successors ( depObj._successors ), _outputObjects( ), _readObjects(), _objectLock() {}

        /*! \brief Assign operator, can be self-assigned.
         *  \param depObj another DependableObject
         */
         const DependableObject & operator= ( const DependableObject &depObj )
         {
            if ( this == &depObj ) return *this; 
            _id = depObj._id;
            _numPredecessors = depObj._numPredecessors;
            _references = depObj._references;
            _successors = depObj._successors;
            _outputObjects = depObj._outputObjects;
            return *this;
         }

        /*! \brief Virtual destructor
         */
         virtual ~DependableObject ( ) { }

         virtual void init ( ) { }

         virtual void dependenciesSatisfied ( ) { }

         virtual void wait ( ) { }

         virtual bool waits ( ) { return false; }

         virtual unsigned long getDescription ( ) { return NULL; }

        /*! \brief Id setter function.
         *         The id will be unique for DependableObjects in the same Dependency Domain.
         *  \param id identifier to be assigned.
         */
         void setId ( unsigned int id )
         {
            _id = id;
         }

        /*! \brief Id getter function.
         *         Returns the id  for the DependableObject (unique in its domain).
         */
         unsigned int getId ()
         {
            return _id;
         }

        /*! \brief Increase the number of predecessors of the DependableObject.
         */
         void increasePredecessors ( )
         {
              _numPredecessors++; 
         }

        /*! \brief Decrease the number of predecessors of the DependableObject
         *         if it becomes 0, the dependencies are satisfied and the virtual
         *         method dependenciesSatisfied is invoked.
         */
         int decreasePredecessors ( )
         {
            int  numPredecessors = --_numPredecessors; 
            if ( numPredecessors == 0 ) {
               dependenciesSatisfied( );
            }
            return numPredecessors;
         }

        /*! \brief Obtain the list of successors
         *  \return List of DependableObject* that depend on "this"
         */
         DependableObjectVector & getSuccessors ( )
         {
            return _successors;
         }

        /*! \brief Add a successor to the successors list
         *  \param depObj DependableObject to be added.
         */
         void addSuccessor ( DependableObject &depObj )
         {
            _successors.push_back ( &depObj );
         }

        /*! \brief Add an output object to the list.
         *  \sa TrackableObject
         */
         void addOutputObject ( TrackableObject *outObj )
         {
            _outputObjects.push_back ( outObj );
         }

        /*! \brief Get the list of output objects.
         *  \sa TrackableObject
         */
         TrackableObjectVector & getOutputObjects ( )
         {
            return _outputObjects;
         }
         
        /*! \brief Add a read object to the list.
         *  \sa TrackableObject
         */
         void addReadObject ( TrackableObject *readObj )
         {
            _readObjects.push_back( readObj );
         }
         
        /*! \brief Get the list of read objects.
         *  \sa TrackableObject
         */
         TrackableObjectVector & getReadObjects ( )
         {
            return _readObjects;
         }

        /*! \brief Increases the object's references counter
         */
         void increaseReferences()
         {
            _references++;
         }

        /*! \brief Get exclusive access to the object
         */
         void lock ( )
         {
            _objectLock.acquire();
            memoryFence();
         }

        /*! \brief Release object's lock
         */
         void unlock ( )
         {
            memoryFence();
            _objectLock.release();
         }

        /*! \brief Dependable Object depObj is finished and its outgoing dependencies are removed.
         *  NOTE: this function is not thread safe
         *  \param desObj Dependable Object that finished
         *  \sa DependableObject
         */
         void finished ( );
   };

};

#endif
