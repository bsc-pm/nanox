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
#include "atomic_decl.hpp"
#include "dependableobject_decl.hpp"
#include "regionstatus_decl.hpp"
#include "dataaccess_decl.hpp"
#include "regiontree_decl.hpp"
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
      private:
         typedef RegionTree<RegionStatus> RegionMap; /**< Maps regions to \a RegionStatus objects */

         static Atomic<int>   _atomicSeed;           /**< ID seed for the domains */
         int                  _id;                   /**< Domain's id */
         unsigned int         _lastDepObjId;         /**< Id to be given to the next submitted DependableObject */
         RegionMap            _regionMap;            /**< Used to track dependencies between DependableObject */
         RecursiveLock        _instanceLock;         /**< Needed to access _regionMap */
         
         static Atomic<int>   _tasksInGraph;         /**< Current number of tasks in the graph */
         static Lock          _lock;

        /*! \brief Finalizes a reduction if active.
         *  \param[in,out] regionStatus status of the subregion
         *  \param region accessed region
         */
         inline void finalizeReduction( RegionStatus &regionStatus, Region const &region );
         
        /*! \brief Makes a DependableObject depend on the last writer of a region.
         *  \param depObj target DependableObject
         *  \param regionStatus status of the region
         */
         inline void dependOnLastWriter( DependableObject &depObj, RegionStatus const &regionStatus );
         
        /*! \brief Makes a DependableObject depend on the the readers of a region and sets it as its last writer.
         *  \param depObj target DependableObject
         *  \param regionStatus status of the region
         */
         inline void dependOnReadersAndSetAsWriter( DependableObject &depObj, RegionStatus &regionStatus, Region const &region );
         
        /*! \brief Makes a DependableObject a reader of a region.
         *  \param depObj target DependableObject
         *  \param regionStatus status of the region
         *  \param region accessed region
         */
         inline void addAsReader( DependableObject &depObj, RegionStatus &regionStatus );
         
        /*! \brief Adds a commutative region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param subregion accessed region
         *  \param accessType kind of region access
         *  \param[in,out] regionStatus status of the subregion
         */
         inline void submitDependableObjectCommutativeDataAccess( DependableObject &depObj, Region const &subregion, AccessType const &accessType, RegionStatus &regionStatus );
         
        /*! \brief Adds an inout region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param subregion accessed region
         *  \param accessType kind of region access
         *  \param[in,out] regionStatus status of the subregion
         */
         inline void submitDependableObjectInoutDataAccess( DependableObject &depObj, Region const &subregion, AccessType const &accessType, RegionStatus &regionStatus );
         
        /*! \brief Adds an input region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param subregion accessed region
         *  \param accessType kind of region access
         *  \param[in,out] regionStatus status of the subregion
         */
         inline void submitDependableObjectInputDataAccess( DependableObject &depObj, Region const &subregion, AccessType const &accessType, RegionStatus &regionStatus );
         
        /*! \brief Adds an output region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param subregion accessed region
         *  \param accessType kind of region access
         *  \param[in,out] regionStatus status of the subregion
         */
         inline void submitDependableObjectOutputDataAccess( DependableObject &depObj, Region const &subregion, AccessType const &accessType, RegionStatus &regionStatus );
         
        /*! \brief Adds a region access of a DependableObject to the domains dependency system.
         *  \param depObj target DependableObject
         *  \param region accessed region
         *  \param accessType kind of region access
         */
         inline void submitDependableObjectDataAccess( DependableObject &depObj, Region const &region, AccessType const &accessType );
         
        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param begin Iterator to the start of the list of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \param end Iterator to the end of the mentioned list.
         *  \sa DataAccess DependableObject TrackableObject
         */
//<<<<<<< HEAD
         template<typename const_iterator>
         void submitDependableObjectInternal ( DependableObject &depObj, const_iterator begin, const_iterator end );
         
//=======
         /* \param callback A function to call when a WD has a successor [Optional].
         *  \sa Dependency DependableObject TrackableObject
         */
//         template<typename iterator>
//         void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end, SchedulePolicySuccessorFunctor* callback );
//
//>>>>>>> cluster
      private:
        /*! \brief DependenciesDomain copy assignment operator (private)
         */
         const DependenciesDomain & operator= ( const DependenciesDomain &depDomain );
      public:

        /*! \brief DependenciesDomain default constructor
         */
         DependenciesDomain ( );

        /*! \brief DependenciesDomain copy constructor
         */
         DependenciesDomain ( const DependenciesDomain &depDomain );

        /*! \brief DependenciesDomain destructor
         */
         ~DependenciesDomain ( );

        /*! \brief get object's id
         */
         int getId ();

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
//<<<<<<< HEAD
         *  \param dataAccesses List of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \sa DataAccess DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject& depObj, std::vector< DataAccess > const &dataAccesses );

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param dataAccesses List of data accesses that determine the dependencies to be associated to the Dependable Object.
         *  \param numDataAccesses Number of data accesses in the list.
         *  \sa DataAccess DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject &depObj, size_t numDataAccesses, DataAccess const *dataAccesses );
//=======
//         *  \param deps List of dependencies to be associated to the Dependable Object.
//         *  \param callback A function to call when a WD has a successor [Optional].
//         *  \sa Dependency DependableObject TrackableObject
//         */
//         void submitDependableObject ( DependableObject &depObj, std::vector<Dependency> &deps, SchedulePolicySuccessorFunctor* callback = NULL );
//
//        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
//         *  \param depObj DependableObject to be added to the domain.
//         *  \param deps List of dependencies to be associated to the Dependable Object.
//         *  \param numDeps Number of dependenices in the list.
//         *  \param callback A function to call when a WD has a successor [Optional].
//         *  \sa Dependency DependableObject TrackableObject
//         */
//         void submitDependableObject ( DependableObject &depObj, size_t numDeps, Dependency* deps, SchedulePolicySuccessorFunctor* callback = NULL );
//>>>>>>> cluster

        /*! \brief Removes the DependableObject from the role of last writer of a region.
         *  \param depObj DependableObject to be stripped of the last writer role
         *  \param region Region that must be affected
         */
         void deleteLastWriter ( DependableObject &depObj, Region const &region );
         
        /*! \brief Removes the DependableObject from the reader list of a region.
         *  \param depObj DependableObject to be removed as a reader
         *  \param region Region that must be affected
         */
         void deleteReader ( DependableObject &depObj, Region const &region );
         
        /*! \brief Removes a CommutableDO from a region.
         *  \param commDO CommutationDO to be removed
         *  \param region Region that must be affected
         */
         void removeCommDO ( CommutationDO *commDO, Region const &region );
         
         static void increaseTasksInGraph();

         static void decreaseTasksInGraph();

        /*! \brief Returns a reference to the instance lock
         */
         RecursiveLock& getInstanceLock();
         
        /*! \brief returns a reference to the static lock
         */
         Lock& getLock();
         
        /*! \brief Get exclusive access to static data
         */
         static void lock ( );

        /*! \brief Release the static lock
         */
         static void unlock ( );
         
         unsigned int getNumReaders( Region const &region );
         unsigned int getNumAllReaders();
         
         void dump(std::string const &function = "", std::string const &file = "", size_t line = 0);
   };

};

#endif

