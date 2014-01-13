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

#ifndef _NANOS_DEPENDABLE_OBJECT_DECL
#define _NANOS_DEPENDABLE_OBJECT_DECL
#include <stdlib.h>
#include <list>
#include <set>
#include <vector>
#include "atomic_decl.hpp"
#include "dataaccess_decl.hpp"
#include "dependenciesdomain_fwd.hpp"
#include "basedependency_fwd.hpp"

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
    *
    *  Form the dependency graph and are representants of the entities that have dependencies. The
    *  DependableObject class provides four virtual methods which are used by the Domain when some
    *  action needs to be taken. The dependenciesSatisfied() method, for instance is invoked when
    *  an object's dependencies are satisfied and the specific implementation of the
    *  DependableObject will make the action desired for this event.
    *
    *  - init(): Is invoked by the DependenciesDomain when the DependableObeject is submitted.
    *    Initializes the DependableObject internal members if necessary.
    *  - wait(): Is invoked by DependenciesDomain when the 'DependableObject has been submitted.
    *  - dependenciesSatisfied(): Invoked by DependenciesDomain when the object's dependencies
    *    are satisfied.
    *  - waits(): Used by the DependenciesDomain to know whether the wait() method waits until
    *    dependencies are
    *  - satisfied. If so, the only action during DependableObject submission is adding it as
    *    successor for the tasks updating its dependencies.
    *
    *  There is two kinds of DependableObject currently which are associated to WorkDescriptors.
    *  DOSubmit represents an entity (task) which is submitted to the dependency system and its
    *  action takes place when its dependencies are satisfied. When a DOWait is submitted to a
    *  domain, the thread executing submitDependableObject() will add the DOWait to the dependency
    *  system and wait until its dependencies are satisfied before returning control to the caller.
    *
    */
   class DependableObject
   {
      public:
         typedef std::set<DependableObject *> DependableObjectVector; /**< Type vector of successors  */
         typedef std::vector<BaseDependency*> TargetVector; /**< Type vector of output objects */
         
      private:
         unsigned int             _id;              /**< DependableObject identifier */
         Atomic<unsigned int>     _numPredecessors; /**< Number of predecessors locking this object */
         unsigned int             _references;      /** References counter */
         DependableObjectVector   _successors;      /**< List of successiors */
         DependenciesDomain      *_domain;          /**< DependenciesDomain where this is located */
         TargetVector             _outputObjects;   /**< List of output objects */
         TargetVector             _readObjects;     /**< List of read objects */
         Lock                     _objectLock;      /**< Lock to do exclusive use of the DependableObject */
         volatile bool            _submitted;

      public:
        /*! \brief DependableObject default constructor
         */
         DependableObject ( ) 
            :  _id ( 0 ), _numPredecessors ( 0 ), _references(1), _successors(), _domain( NULL ), _outputObjects(),
               _readObjects(), _objectLock(), _submitted(false) {}
        /*! \brief DependableObject copy constructor
         *  \param depObj another DependableObject
         */
         DependableObject ( const DependableObject &depObj )
            : _id ( depObj._id ), _numPredecessors ( depObj._numPredecessors ), _references(depObj._references),
              _successors ( depObj._successors ), _domain ( depObj._domain ), _outputObjects( ), _readObjects(), _objectLock(), _submitted(false) {}

        /*! \brief DependableObject copy assignment operator, can be self-assigned.
         *  \param depObj another DependableObject
         */
         const DependableObject & operator= ( const DependableObject &depObj );

        /*! \brief DependableObject virtual destructor
         */
         virtual ~DependableObject ( );

        // FIXME DOC NEEDED FOR THIS FUNCTIONS
         virtual void init ( ) { }

         virtual void dependenciesSatisfied ( ) { }

         /*! \brief Waits until the dependencies with the given addresses
          * are satisfied.
          * \param flushDeps List of memory addresses.
          * \note Previously we were passing a list of Dependency pointers,
          * and used getDepAddress to create the flushDeps list inside this 
          * function. As the regions code passes regions and does the same,
          * the list will have to be created before calling this function.
          * The regions code will call Region::getFirstValue() and the
          * non region code will use the address + the offset.
          */
         virtual void wait ( std::list<uint64_t>const & flushDeps  ) { }

         virtual bool waits ( );

         virtual unsigned long getDescription ( );

         /*! \brief Get the related object which actually has the dependence
          */
         virtual void * getRelatedObject ( );
         
         /*! \brief Get the related object which actually has the dependence
          * (const version)
          */
         virtual const void * getRelatedObject ( ) const;

         /*! \brief Instrument predecessor -> successor dependency
          */
         virtual void instrument ( DependableObject& successor ) { }

        /*! \brief Id setter function.
         *         The id will be unique for DependableObjects in the same Dependency Domain.
         *  \param id identifier to be assigned.
         */
         void setId ( unsigned int id );

        /*! \brief Id getter function.
         *         Returns the id  for the DependableObject (unique in its domain).
         */
         unsigned int getId () const;

        /*! \brief Increase the number of predecessors of the DependableObject.
         */
         int increasePredecessors ( );

        /*! \brief Decrease the number of predecessors of the DependableObject
         *         if it becomes 0, the dependencies are satisfied and the virtual
         *         method dependenciesSatisfied is invoked.
         */
         int decreasePredecessors ( );

         /*! \brief  Returns the number of predecessors of this DependableObject
          */
         int numPredecessors () const;

        /*! \brief Obtain the list of successors
         *  \return List of DependableObject* that depend on "this"
         */
         DependableObjectVector & getSuccessors ( );

        /*! \brief Add a successor to the successors list
         *  \param depObj DependableObject to be added.
         *  returns true if the successor didn't already exist in the list (a new edge has been added)
         */
         bool addSuccessor ( DependableObject &depObj );
         
        /*! \brief Get the DependenciesDomain where this belongs
         *  \returns the DependenciesDomain where this belongs
         */
         DependenciesDomain * getDependenciesDomain ( ) const;

        /*! \brief Set the DependenciesDomain where this belongs
         *  \param dependenciesDomain the DependenciesDomain where this belongs
         */
         void setDependenciesDomain ( DependenciesDomain *dependenciesDomain );

        /*! \brief Add an output object to the list.
         *  \sa TrackableObject
         */
         void addWriteTarget ( BaseDependency const &outObj );

        /*! \brief Get the list of output objects.
         *  \sa TrackableObject
         */
         TargetVector const & getWrittenTargets ( );
         
        /*! \brief Add a read object to the list.
         *  \sa TrackableObject
         */
         void addReadTarget ( BaseDependency const &readObj );
         
        /*! \brief Get the list of read objects.
         *  \sa TrackableObject
         */
         TargetVector const & getReadTargets ( );

        /*! \brief Increases the object's references counter
         */
         void increaseReferences();

        /*! \brief Sets references to 1 as the object had just been initialized
         */
         void resetReferences();

        /*! \brief returns true if the DependableObject has been submitted in the domain
         */
         bool isSubmitted();

        /*! \breif sets the DO to submitted state
         */
         void submitted();

        /*! \brief returns a reference to the object's lock
         */
         Lock& getLock();

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
