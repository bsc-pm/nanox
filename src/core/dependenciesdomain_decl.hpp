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

#ifndef _NANOS_DEPENDENCIES_DOMAIN_DECL
#define _NANOS_DEPENDENCIES_DOMAIN_DECL
#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include <string>
#include "atomic_decl.hpp"
#include "dependableobject_decl.hpp"
#include "trackableobject_decl.hpp"
//#include "regionstatus_decl.hpp"
#include "dependency_decl.hpp"
#include "compatibility.hpp"
#include "schedule_fwd.hpp"


namespace nanos
{

   namespace dependencies_domain_internal {
      class AccessType;
   }


   using namespace dependencies_domain_internal;

  /*! \class DependenciesDomain
   *  \brief Each domain is an independent context in which dependencies between DependableObject are managed
   */
   class DependenciesDomain
   {
      public:
         typedef TrackableObject MappedType;
         //! In the regions version, this would be Region.
         typedef void* Target;
      private:
         //typedef TR1::unordered_map<void *, TrackableObject*> DepsMap; /**< Maps addresses to Trackable objects */
         typedef TR1::unordered_map<Target, MappedType*> DepsMap; /**< Maps addresses to Trackable objects */

         static Atomic<int>   _atomicSeed;           /**< ID seed for the domains */
         int                  _id;                   /**< Domain's id */
         unsigned int         _lastDepObjId;         /**< Id to be given to the next submitted DependableObject */
         DepsMap              _addressDependencyMap; /**< Used to track dependencies between DependableObject */
         RecursiveLock        _instanceLock;         /**< Needed to access _addressDependencyMap */
         static Atomic<int>   _tasksInGraph;         /**< Current number of tasks in the graph */
         static Lock          _lock;
         
         /*! \brief Finalizes a reduction if active.
          *  \param[in,out] trackableObject status of the target.
         *  \param target accessed memory address
          */
         inline void finalizeReduction( MappedType &trackableObject, const Target& target );
         
        /*! \brief Makes a DependableObject depend on the last writer of a region.
         *  \param depObj target DependableObject
         *  \param status status of the address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void dependOnLastWriter( DependableObject &depObj, MappedType const &status, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Makes a DependableObject depend on the the readers of a region and sets it as its last writer.
         *  \param depObj target DependableObject
         *  \param status status of the address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void dependOnReadersAndSetAsWriter( DependableObject &depObj, MappedType &status, Target const &target, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Makes a DependableObject a reader of a region.
         *  \param depObj target DependableObject
         *  \param status status of the address
         *  \param target accessed base address
         */
         inline void addAsReader( DependableObject &depObj, MappedType &status );
         
        /*! \brief Adds a commutative region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param target accessed base address
         *  \param accessType kind of region access
         *  \param[in,out] status status of the base address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void submitDependableObjectCommutativeDataAccess( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Adds an inout region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param target accessed memory address
         *  \param accessType kind of region access
         *  \param[in,out] status status of the base address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void submitDependableObjectInoutDataAccess( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Adds an input region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param target accessed memory address
         *  \param accessType kind of region access
         *  \param[in,out] status status of the base address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void submitDependableObjectInputDataAccess( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Adds an output region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param target accessed memory address
         *  \param accessType kind of region access
         *  \param[in,out] status status of the base address
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void submitDependableObjectOutputDataAccess( DependableObject &depObj, Target const &target, AccessType const &accessType, MappedType &status, SchedulePolicySuccessorFunctor* callback );
         
        /*! \brief Adds a region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param target accessed memory address
         *  \param accessType kind of region access
         *  \param callback Function to call if an immediate predecessor is found.
         */
         inline void submitDependableObjectDataAccess( DependableObject &depObj, Target const &target, AccessType const &accessType, SchedulePolicySuccessorFunctor* callback );

        /*! \brief Looks for the dependency's address in the domain and returns the trackableObject associated.
         *  \param dep Dependency to be checked.
         *  \sa Dependency TrackableObject
         */
         MappedType* lookupDependency ( const Target &target );
        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param begin Iterator to the start of the list of dependencies to be associated to the Dependable Object.
         *  \param end Iterator to the end of the mentioned list.
         *  \param callback A function to call when a WD has a successor [Optional].
         *  \sa Dependency DependableObject TrackableObject
         */
         template<typename iterator>
         void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end, SchedulePolicySuccessorFunctor* callback );

      private:
        /*! \brief DependenciesDomain copy assignment operator (private)
         */
         const DependenciesDomain & operator= ( const DependenciesDomain &depDomain );
      public:

        /*! \brief DependenciesDomain default constructor
         */
         DependenciesDomain ( ) :  _id( _atomicSeed++ ), _lastDepObjId ( 0 ), _addressDependencyMap( ) {}

        /*! \brief DependenciesDomain copy constructor
         */
         DependenciesDomain ( const DependenciesDomain &depDomain )
            : _id( _atomicSeed++ ), _lastDepObjId ( depDomain._lastDepObjId ),
              _addressDependencyMap ( depDomain._addressDependencyMap ) {}

        /*! \brief DependenciesDomain destructor
         */
         ~DependenciesDomain ( );

        /*! \brief get object's id
         */
         int getId ();

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param deps List of dependencies to be associated to the Dependable Object.
         *  \param callback A function to call when a WD has a successor [Optional].
         *  \sa Dependency DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject &depObj, std::vector<Dependency> &deps, SchedulePolicySuccessorFunctor* callback = NULL );

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param deps List of dependencies to be associated to the Dependable Object.
         *  \param numDeps Number of dependenices in the list.
         *  \param callback A function to call when a WD has a successor [Optional].
         *  \sa Dependency DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject &depObj, size_t numDeps, Dependency* deps, SchedulePolicySuccessorFunctor* callback = NULL );

         /*! \brief Removes the DependableObject from the role of last writer of a region.
         *  \param depObj DependableObject to be stripped of the last writer role
         *  \param target Address/region that must be affected
         */
         void deleteLastWriter ( DependableObject &depObj, Target const &target );
         
        /*! \brief Removes the DependableObject from the reader list of a region.
         *  \param depObj DependableObject to be removed as a reader
         *  \param target Address/region that must be affected
         */
         void deleteReader ( DependableObject &depObj, Target const &target );

        /*! \brief Removes a CommutableDO from a region.
         *  \param commDO CommutationDO to be removed
         *  \param target Address/region that must be affected
         */
         void removeCommDO ( CommutationDO *commDO, Target const &target );

         static void increaseTasksInGraph();

         static void decreaseTasksInGraph();

        /*! \brief Returns a reference to the instance lock
         */
         RecursiveLock& getInstanceLock();
         
        /*! \brief returns a reference to the static lock
         */
         Lock& getLock();

        /*! \brief Get exclusive access to the object
         */
         static void lock ( );

        /*! \brief Release object's lock
         */
         static void unlock ( );
   };
   
   /*! \class DependenciesManager
    * \brief Interface class of plugins used for dependencies domain.
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
   };

};

#endif

