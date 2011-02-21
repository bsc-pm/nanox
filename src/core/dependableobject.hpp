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
#include <set>
#include <vector>
#include "atomic.hpp"
#include "trackableobject_fwd.hpp"
#include "dependency.hpp"

namespace nanos
{
   class DependableObject;

   class DependableObjectPredicate
   {
      public:
         DependableObjectPredicate() {}
         virtual ~DependableObjectPredicate() {}

         virtual bool operator() (DependableObject &obj) = 0;
   };


   /*! \class DependableObject
    *  \brief Abstract entity submitted to the Dependency system
    */
   class DependableObject
   {
      public:
         typedef std::set<DependableObject *> DependableObjectVector; /**< Type vector of successors  */
         typedef std::vector<TrackableObject *> TrackableObjectVector; /**< Type vector of output objects */
         
      private:
         unsigned int             _id;              /**< DependableObject identifier */
         Atomic<unsigned int>     _numPredecessors; /**< Number of predecessors locking this object */
         unsigned int             _references;      /** References counter */
         DependableObjectVector   _successors;      /**< List of successiors */
         TrackableObjectVector    _outputObjects;   /**< List of output objects */
         TrackableObjectVector    _readObjects;     /**< List of read objects */
         Lock                     _objectLock;      /**< Lock to do exclusive use of the DependableObject */

      public:
        /*! \brief DependableObject default constructor
         */
         DependableObject ( ) 
            :  _id ( 0 ), _numPredecessors ( 0 ), _references(1), _successors(), _outputObjects(),
               _readObjects(), _objectLock() {}
        /*! \brief DependableObject copy constructor
         *  \param depObj another DependableObject
         */
         DependableObject ( const DependableObject &depObj )
            : _id ( depObj._id ), _numPredecessors ( depObj._numPredecessors ), _references(depObj._references),
              _successors ( depObj._successors ), _outputObjects( ), _readObjects(), _objectLock() {}
        /*! \brief DependableObject copy assignment operator, can be self-assigned.
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
        /*! \brief DependableObject virtual destructor
         */
         virtual ~DependableObject ( ) { }

         virtual void init ( ) { }

         virtual void dependenciesSatisfied ( ) { }

         virtual void wait ( std::list<Dependency *> deps ) { }

         virtual bool waits ( ) { return false; }

         virtual unsigned long getDescription ( ) { return 0; }

         /*! \brief Get the related object which actually has the dependence
          */
         virtual void * getRelatedObject ( ) { return NULL; }

         /*! \brief Instrument predecessor -> successor dependency
          */
         virtual void instrument ( DependableObject& successor ) { }

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
         int increasePredecessors ( )
         {
              return _numPredecessors++; 
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

         /*! \brief  Returns the number of predecessors of this DependableObject
          */
         int numPredecessors () const { return _numPredecessors.value(); }

        /*! \brief Obtain the list of successors
         *  \return List of DependableObject* that depend on "this"
         */
         DependableObjectVector & getSuccessors ( )
         {
            return _successors;
         }

        /*! \brief Add a successor to the successors list
         *  \param depObj DependableObject to be added.
         *  returns true if the successor didn't already exist in the list (a new edge has been added)
         */
         bool addSuccessor ( DependableObject &depObj )
         {
            return _successors.insert ( &depObj ).second;
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

        /*! \brief Sets references to 1 as the object had just been initialized
         */
         void resetReferences()
         {
            _references = 1;
         }

	/*! \brief returns a reference to the object's lock
         */
         Lock& getLock()
         {
            return _objectLock;
         }
        /*! \brief Dependable Object depObj is finished and its outgoing dependencies are removed.
         *  NOTE: this function is not thread safe
         *  \param desObj Dependable Object that finished
         *  \sa DependableObject
         */
         void finished ( );

        /*! If there is an object that only depends from this dependable object, then release it and
            return it
         */
         DependableObject * releaseImmediateSuccessor ( DependableObjectPredicate &condition );
         
   };

};

#endif
