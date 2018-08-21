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

#ifndef _NANOS_DEPENDABLE_OBJECT_DECL
#define _NANOS_DEPENDABLE_OBJECT_DECL
#include <stdlib.h>
#include <list>
#include <set>
#include <vector>
#include <stdint.h>

#include "atomic_decl.hpp"
#include "lock_decl.hpp"

#include "dependenciesdomain_fwd.hpp"
#include "basedependency_fwd.hpp"
#include "workdescriptor_fwd.hpp"

#include "dataaccess_decl.hpp"

namespace nanos {

   class DependableObject;

   class DOSchedulerData
   {
      public:
         DOSchedulerData() {}
         virtual ~DOSchedulerData() {}
         virtual void reset() = 0;
   };


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
         typedef std::pair< unsigned int, DependableObject * > DependableObjectVectorKey;
         typedef std::set<DependableObjectVectorKey> DependableObjectVector; /**< Type vector of successors  */
         typedef std::vector<BaseDependency*> TargetVector; /**< Type vector of output objects */
         
      private:
         unsigned int             _id;              /**< DependableObject identifier */
         Atomic<unsigned int>     _numPredecessors; /**< Number of predecessors locking this object */
         unsigned int             _references;      /** References counter */
         DependableObjectVector   _predecessors;    /**< List of predecessors */
         DependableObjectVector   _successors;      /**< List of successors */
         DependenciesDomain      *_domain;          /**< DependenciesDomain where this is located */
         TargetVector             _outputObjects;   /**< List of output objects */
         TargetVector             _readObjects;     /**< List of read objects */
         mutable Lock             _objectLock;      /**< Lock to do exclusive use of the DependableObject */
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         bool                     _submitted;
#else
         volatile bool            _submitted;
#endif
         bool                     _needsSubmission; /**< Does this DependableObject need to be submitted? */
         WorkDescriptor           *_wd;             /**< Pointer to the work descriptor represented by this DependableObject */
         DOSchedulerData          *_schedulerData;  /**< Data needed for specific scheduling policies */
         int _num;
         int _lss;

      public:
        /*! \brief DependableObject default constructor
         */
         DependableObject ( ) 
            :  _id ( 0 ), _numPredecessors ( 0 ), _references( 1 ), _predecessors(), _successors(), _domain( NULL ), _outputObjects(),
               _readObjects(), _objectLock(), _submitted( false ), _needsSubmission( false ), _wd( NULL ), _schedulerData(NULL), _num(0), _lss(-1) {}

         DependableObject ( WorkDescriptor *wd ) 
            :  _id ( 0 ), _numPredecessors ( 0 ), _references( 1 ), _predecessors(), _successors(), _domain( NULL ), _outputObjects(),
               _readObjects(), _objectLock(), _submitted( false ), _needsSubmission( false ), _wd( wd ), _schedulerData(NULL), _num(0), _lss(-1) {}

        /*! \brief DependableObject copy constructor
         *  \param depObj another DependableObject
         */
         DependableObject ( const DependableObject &depObj );

        /*! \brief DependableObject copy assignment operator, can be self-assigned.
         *  \param depObj another DependableObject
         */
         const DependableObject & operator= ( const DependableObject &depObj );

        /*! \brief DependableObject virtual destructor
         */
         virtual ~DependableObject ( );

        // FIXME DOC NEEDED FOR THIS FUNCTIONS
         virtual void init ( ) { }

         /*! \brief This method is automatically called when
          *  decreasePredecessors reduces the number to 0.
          *  \note As there is a batch-release of dependencies, this method must
          *  call dependenciesSatisfiedNoSubmit() to avoid code duplication.
          *  \see decreasePredecessors(), dependenciesSatisfiedNoSubmit()
          */
         virtual void dependenciesSatisfied ( ) { }
         
         /*! \brief Because of the batch-release mechanism,
          *  dependenciesSatisfied will never be called, so common code
          *  between the normal path and the batch path must go here.
          *  \see dependenciesSatisfied()
          */
         virtual void dependenciesSatisfiedNoSubmit ( ) { }

         virtual bool waits ( );

         virtual unsigned long getDescription ( );

         /*! \brief Get the related object which actually has the dependence
          */
         virtual void * getRelatedObject ( );
         
         /*! \brief Get the related object which actually has the dependence
          * (const version)
          */
         virtual const void * getRelatedObject ( ) const;
         
         /*! \brief Checks if this class supports batch release and, if so, if
          *  it has only one predecessor.
          */
         virtual bool canBeBatchReleased ( ) const;

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
         *         method dependenciesSatisfied is invoked. It can be also a blocking
         *         call in some cases, if blocking is set to true.
         */
         virtual int decreasePredecessors ( std::list<uint64_t> const * flushDeps, DependableObject * finishedPred,
               bool batchRelease, bool blocking = false );

         /*! \brief Auxiliar function of decreasePredecessors() that encapsulates the
          *         mutual exclusion operations.
          */
          virtual void decreasePredecessorsInLock (  DependableObject * finishedPred, int numPred );

         /*! \brief  Returns the number of predecessors of this DependableObject
          */
         int numPredecessors () const;

         /*! \brief Obtain the list of predecessors
          *  \return List of DependableObject* that "this" depends on
          */
         DependableObjectVector & getPredecessors ( );

        /*! \brief Obtain the list of successors
         *  \return List of DependableObject* that depend on "this"
         */
         DependableObjectVector & getSuccessors ( );

         /*! \brief Add a predecessor to the predecessors list
          *  \param depObj DependableObject to be added.
          *  returns true if the predecessor didn't already exist in the list (a new edge has been added)
          */
         bool addPredecessor ( DependableObject &depObj );

        /*! \brief Add a successor to the successors list
         *  \param depObj DependableObject to be added.
         *  returns true if the successor didn't already exist in the list (a new edge has been added)
         */
         bool addSuccessor ( DependableObject &depObj );
         
         /*! \brief Delete a successor from the successors list
          *  \param depObj DependableObject to be erased.
          *  returns true if the successor was found (and consequently erased)
          */
         bool deleteSuccessor ( DependableObject *depObj );
         bool deleteSuccessor ( DependableObject &depObj );


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

        /*! \brief sets the DO to submitted state
         */
         void submitted();

         /*! \brief returns true if the DependableObject needs to be submitted in the domain
          */
         bool needsSubmission() const;

         /*! \brief sets the DO to submitted
          */
         void enableSubmission();

         /*! \brief sets the DO to not submitted
          */
         void disableSubmission();

         /*! \brief returns a reference to the object's lock
         */
         Lock& getLock();

        /*! \brief Dependable Object depObj is finished and its outgoing dependencies are removed.
         *  NOTE: this function is not thread safe
         *  \param desObj Dependable Object that finished
         *  \sa DependableObject
         */
         void finished ( );
         
         
         
        /*! \brief Release input dependencies
         *  NOTE: this function is not thread safe
         */
         void releaseReadDependencies ();

        /*! If there is an object that only depends from this dependable object, then release it and
            return it
         */
         DependableObject * releaseImmediateSuccessor ( DependableObjectPredicate &condition, bool keepDeps );

         void setWD( WorkDescriptor *wd );
         WorkDescriptor * getWD( void ) const;

         DOSchedulerData* getSchedulerData ( );
         void setSchedulerData ( DOSchedulerData * scData );

         int getNum() const { return _num; };
         int getLSS() const { return _lss; };
   };

} // namespace nanos

#endif
