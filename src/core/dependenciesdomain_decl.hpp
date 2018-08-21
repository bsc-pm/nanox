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

//! \file dependenciesdomain_decl.hpp
//! \brief Dependencies main classes declaration.
//
//! \defgroup core_dependencies Dependencies module
//! \ingroup core

/*!\page    core_dependencies
 * \ingroup core_dependencies
 *
 * \section dependencies_domain Dependencies Domain
 * \copydoc nanos::DependenciesDomain
 *
 * \section data_access Data Access Item
 * \copydoc nanos::DataAccess
 *
 * \section dependable_object Dependable Object
 * \copydoc nanos::DependableObject
 *
 * \section wd_and_deps WorkDescriptors and DependenciesDomain
 *
 * Each WorkDescriptor has a DependenciesDomain in which DependableObjects can be submitted.
 * WorkDescriptors are submitted to the dependency system of the parent's DependenciesDomain
 * using their associated DOSubmit object. The DOSubmit object encapsulates the actions to
 * take when the dependencies of the submitted WorkDescriptor are satisfied.
 *
 * When a WD needs to be synchronized with some of its children, its DOWait object is
 * submitted to its own domain with the given dependencies and will wait until that
 * dependencies are satisfied.
 *
 */
#ifndef _NANOS_DEPENDENCIES_DOMAIN_DECL
#define _NANOS_DEPENDENCIES_DOMAIN_DECL
#include "atomic_decl.hpp"
#include "recursivelock_decl.hpp"
#include "lock_decl.hpp"
#include "dataaccess_decl.hpp"
#include "basedependency_decl.hpp"

#include "commutationdepobj_fwd.hpp"
#include "dependableobject_fwd.hpp"
#include "schedule_fwd.hpp"

#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include <string>

namespace nanos {

using namespace dependencies_domain_internal;
   
  /*! \class DependenciesDomain
   *  Interface class of plugins used for dependencies domain.
   *  \brief Each domain is an independent context in which dependencies between DependableObject are managed
   *
   *  Represents an independent domain to which objects with dependencies (DependableObjects) can be
   *  submitted. WorkDescriptors have associated a DependenciesDomain to which they submit objetcs
   *  representing new tasks with dependencies or a set of dependencies that need to be synchronized
   *  (wait until the dependencies are satisfied).
   *
   *  When a DependableObject is added to the domain, its dependencies are checked using their address as an
   *  identifier. TrackableObjects represent the dynamic status of those dependencies keeping track of which
   *  objects are about to read or write the address they represent. At the end of this process the virtual
   *  method wait() of the DependableObject is invoked. This mechanism allows having different behaviors for
   *  DependableObjects which are transparent to the domain.
   *
   *  When a DependableObject ends, finalizeObject() is invoked in the Domain so that the status of the
   *  dependencies and the TrackableObjects are updated. If, in this process, a DependableObject gets all
   *  its dependencies satisfied, has no more predecessors, the dependenciesSatisfied() method is invoked.
   *
   */
   class DependenciesDomain
   {
      private:

         static Atomic<int>   _atomicSeed;           /**< ID seed for the domains */
         int                  _id;                   /**< Domain's id */
         RecursiveLock        _instanceLock;         /**< Needed to access _addressDependencyMap */
         static Atomic<int>   _tasksInGraph;         /**< Current number of tasks in the graph */
         static Lock          _lock;

      private:
        /*! \brief DependenciesDomain copy assignment operator (private)
         */
         const DependenciesDomain & operator= ( const DependenciesDomain &depDomain );
      public:
        /*! \brief DependenciesDomain default constructor
         */
         DependenciesDomain ( ) :  _id( _atomicSeed++ ) {}

        /*! \brief DependenciesDomain copy constructor
         */
         DependenciesDomain ( const DependenciesDomain &depDomain )
            : _id( _atomicSeed++ ) {}

        /*! \brief DependenciesDomain destructor
         */
         virtual ~DependenciesDomain ( );

        /*! \brief get object's id
         */
         int getId ();
         
         
         /*! \name Main methods to be implemented.
          *  \{
          */
         
         /*! \brief Removes the DependableObject from the role of last writer of a region.
          *  \param depObj DependableObject to be stripped of the last writer role
          *  \param target Address/region that must be affected
          */
         virtual void deleteLastWriter ( DependableObject &depObj, BaseDependency const &target ) = 0;
         
         /*! \brief Removes the DependableObject from the reader list of a region.
          *  \param depObj DependableObject to be removed as a reader
          *  \param target Address/region that must be affected
          */
         virtual void deleteReader ( DependableObject &depObj, BaseDependency const &target ) = 0;
         
         /*! \brief Removes a CommutableDO from a region.
          *  \param commDO CommutationDO to be removed
          *  \param target Address/region that must be affected
          */
         virtual void removeCommDO ( CommutationDO *commDO, BaseDependency const &target ) = 0;

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param dataAccesses List of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \param callback A function to call when a WD has a successor [Optional].
         *  \sa DataAccess DependableObject TrackableObject
         */
         virtual void submitDependableObject ( DependableObject &depObj, std::vector<DataAccess> &dataAccesses, SchedulePolicySuccessorFunctor* callback = NULL ) = 0;

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param dataAccesses List of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \param numDataAccesses List of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \param callback A function to call when a WD has a successor [Optional].
         *  \sa DataAccess DependableObject TrackableObject
         */
         virtual void submitDependableObject ( DependableObject &depObj, size_t numDataAccesses, DataAccess* dataAccesses, SchedulePolicySuccessorFunctor* callback = NULL ) = 0;
         
         /*! \}
          */

         static void increaseTasksInGraph( size_t num = 1 );

         static void decreaseTasksInGraph( size_t num = 1 );

        /*! \brief Returns a reference to the instance lock
         */
         RecursiveLock& getInstanceLock();
         
        /*! \brief returns a reference to the static lock
         */
         Lock& getLock();

        /*! \brief Get exclusive access to the object
         */
         static void lock ( );

        /*! \brief Release the static lock
         */
         static void unlock ( );

        /*! \brief Returns if a given memory address has any pendant write
         *
         *  \param [in] addr memory address
         *  \return if there are any pendant write on the address passed as a parameter
         */
         virtual bool haveDependencePendantWrites ( void *addr ) ;

         //! \brief Finalize all pendant reductions
         virtual void finalizeAllReductions ( void ) ;

         //! \brief Clear all pendants references
         virtual void clearDependenciesDomain ( void ) ;
   };
   
   /*! \class DependenciesManager.
    *  \brief This class is to be implemented by the plugin in order to provide
    *  a method to create DependenciesDomains.
    */
   class DependenciesManager
   {
      private:
         std::string _name;
      private:
         /*! \brief DependenciesManager default constructor (private)
          */
         DependenciesManager ();
         /*! \brief DependenciesManager copy constructor (private)
          */
         DependenciesManager ( DependenciesManager &sp );
         /*! \brief DependenciesManager copy assignment operator (private)
          */
         DependenciesManager& operator= ( DependenciesManager &sp );
         
      public:
         /*! \brief DependenciesManager constructor - with std::string &name
          */
         DependenciesManager ( const std::string &name ) : _name(name) {}
         /*! \brief DependenciesManager constructor - with char *name
          */
         DependenciesManager ( const char *name ) : _name(name) {}
         /*! \brief DependenciesManager destructor
          */
         virtual ~DependenciesManager () {};

         const std::string & getName () const;
         
         /*! \brief Creates a  dependencies domain. To be implemented by the
          *  plugin class.
          */
         virtual DependenciesDomain* createDependenciesDomain () const = 0;
   };
   
} // namespace nanos

#endif

