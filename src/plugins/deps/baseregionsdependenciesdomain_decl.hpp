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

#ifndef _NANOS_BASE_REGIONS_DEPENDENCIES_DOMAIN_DECL
#define _NANOS_BASE_REGIONS_DEPENDENCIES_DOMAIN_DECL

#include "basedependenciesdomain_decl.hpp"


namespace nanos {

   using namespace dependencies_domain_internal;

   /*! \brief Class with common code from the regions and non-regions plugins.
    */
   class BaseRegionsDependenciesDomain : public BaseDependenciesDomain
   {
      protected:         
         /*! \brief Finalizes a reduction if active.
          *  \tparam CONTAINER_T type of the TrackableObject* container
          *  \param[in,out] statusContainer container with the statuses of the addresses/regions
          *  \param target accessed memory address/region
          */
         template<typename CONTAINER_T>
         inline void finalizeReduction( CONTAINER_T &statusContainer, const BaseDependency& target );
         
         /*! \brief Makes a DependableObject depend on the last writer of a region.
          *  \tparam CONTAINER_T type of the TrackableObject* container
          *  \param depObj target DependableObject
          *  \param[in,out] statusContainer container with the statuses of the addresses/regions
          *  \param callback Function to call if an immediate predecessor is found.
          *  \param accessType Current data access type (in, out, inout, concurrent,...)
          */
         template<typename CONTAINER_T>
         inline void dependOnLastWriter( DependableObject &depObj, CONTAINER_T &statusContainer, BaseDependency const &target,
                                         SchedulePolicySuccessorFunctor* callback, AccessType const &accessType );
         
         /*! \brief Makes a DependableObject depend on the the readers of a set of regions.
          *  \tparam CONTAINER_T type of the TrackableObject* container
          *  \param depObj target DependableObject
          *  \param[in] statusContainer container with the statuses of the addresses/regions
          *  \param target accessed base address/region
          *  \param callback Function to call if an immediate predecessor is found.
          *  \param accessType Current data access type (in, out, inout, concurrent,...)
          */
         template<typename CONTAINER_T>
         inline void dependOnReaders( DependableObject &depObj, CONTAINER_T &statusContainer, BaseDependency const &target,
                                      SchedulePolicySuccessorFunctor* callback, AccessType const &accessType );
         
         //! \brief Create a CommutationDO object.
         //! \param[in] target
         //! \param[in] accessType
         //! \param[in] status trackable object
         //! \returns a CommutationDO object
         virtual CommutationDO *createCommutationDO(BaseDependency const &target, AccessType const &accessType, TrackableObject &status);

         /*! \brief Adds a commutative access of a DependableObject to the domains dependency system.
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          */
         template <typename SOURCE_STATUS_T>
         inline CommutationDO *setUpInitialCommutationDependableObject( BaseDependency const &target, AccessType const &accessType, SOURCE_STATUS_T &sourceStatus, TrackableObject &targetStatus );
         
         /*! \brief Adds a commutative access of a DependableObject to the domains dependency system.
          *  \tparam SOURCE_STATUS_T type of the sourceStatus. Can be a container of TrackableObject* or a TrackableObject&
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         template <typename SOURCE_STATUS_T>
         inline void submitDependableObjectCommutativeDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, SOURCE_STATUS_T &sourceStatus, TrackableObject &targetStatus, SchedulePolicySuccessorFunctor* callback );

         /*! \brief Adds an inout access of a DependableObject to the domains dependency system.
          *  \tparam SOURCE_STATUS_T type of the sourceStatus. Can be a container of TrackableObject* or a TrackableObject&
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         template <typename SOURCE_STATUS_T>
         inline void submitDependableObjectInoutDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, SOURCE_STATUS_T &sourceStatus, TrackableObject &targetStatus, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an output access of a DependableObject to the domains dependency system.
          *  \tparam SOURCE_STATUS_T type of the sourceStatus. Can be a container of TrackableObject* or a TrackableObject&
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         template <typename SOURCE_STATUS_T>
         inline void submitDependableObjectOutputDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, SOURCE_STATUS_T &sourceStatus, TrackableObject &targetStatus, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an input region access of a DependableObject to the domains dependency system. 
          *  \tparam SOURCE_STATUS_T type of the sourceStatus. Can be a container of TrackableObject* or a TrackableObject&
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         template <typename SOURCE_STATUS_T>
         inline void submitDependableObjectInputDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, SOURCE_STATUS_T &sourceStatus, TrackableObject &targetStatus, SchedulePolicySuccessorFunctor* callback );
         
      public:
         BaseRegionsDependenciesDomain ( ) :  BaseDependenciesDomain() {}
         
         BaseRegionsDependenciesDomain ( const BaseRegionsDependenciesDomain &depDomain )
            : BaseDependenciesDomain( depDomain ) {}
   };

} // namespace nanos

#endif

