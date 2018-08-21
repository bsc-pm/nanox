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

#ifndef _NANOS_BASE_DEPENDENCIES_DOMAIN_DECL
#define _NANOS_BASE_DEPENDENCIES_DOMAIN_DECL

#include "dependenciesdomain_decl.hpp"
#include "instrumentation_decl.hpp"
#include "system.hpp"

#include "trackableobject_fwd.hpp"

namespace nanos {

   using namespace dependencies_domain_internal;

   /*! \brief Class with common code from the regions and non-regions plugins.
    */
   class BaseDependenciesDomain : public DependenciesDomain
   {
      protected:
         unsigned int _lastDepObjId;         /**< Id to be given to the next submitted DependableObject */
      private:
         /*! \brief Creates a CommutationDO and attaches it to the trackable object.
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the base address
          */
         virtual CommutationDO *createCommutationDO(BaseDependency const &target, AccessType const &accessType, TrackableObject &status);
         NANOS_INSTRUMENT ( nanos_event_key_t   _insKeyDeps[3]; ) /**< Instrumentation key dependences */

      protected:         
         /*! \brief Finalizes a reduction if active.
          *  \param[in,out] trackableObject status of the address/region
          *  \param target accessed memory address/region
          */
         inline void finalizeReduction( TrackableObject &trackableObject, const BaseDependency& target );
         
         /*! \brief Makes a DependableObject depend on the last writer of a region.
          *  \param depObj target DependableObject
          *  \param[in,out] status status of the address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void dependOnLastWriter( DependableObject &depObj, TrackableObject const &status, BaseDependency const &target,
                                          SchedulePolicySuccessorFunctor* callback, AccessType const &accessType );
         
         /*! \brief Makes a DependableObject depend on the the readers of a set of regions.
          *  \param depObj target DependableObject
          *  \param[in] status status of the address/region
          *  \param target accessed base address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void dependOnReaders( DependableObject &depObj, TrackableObject &status, BaseDependency const &target,
                                      SchedulePolicySuccessorFunctor* callback, AccessType const &accessType );
         
         /*! \brief Sets the last writer DependableObject of a region.
          *  \param depObj target DependableObject
          *  \param[in,out] status status of the address
          *  \param target accessed base address/region
          */
         inline void setAsWriter( DependableObject &depObj, TrackableObject &status, BaseDependency const &target );
         
         /*! \brief Makes a DependableObject depend on the the readers of a region and sets it as its last writer.
          *  \param depObj target DependableObject
          *  \param[in,out] status status of the address
          *  \param target accessed base address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void dependOnReadersAndSetAsWriter( DependableObject &depObj, TrackableObject &status, BaseDependency const &target, SchedulePolicySuccessorFunctor* callback, AccessType const &accessType );
         
         /*! \brief Makes a DependableObject a reader of a region/address.
          *  \param depObj target DependableObject
          *  \param[in,out] status status of the address/region
          */
         inline void addAsReader( DependableObject &depObj, TrackableObject &status );
       
         /*! \brief Sets up a target Commutation Dependable Object for a Commutation access with a fake additional dependency.
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the address/region
          */
         inline CommutationDO *setUpTargetCommutationDependableObject( BaseDependency const &target, AccessType const &accessType, TrackableObject &status );
         
         /*! \brief Adds a commutative access of a DependableObject to the domains dependency system.
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the address/region
          */
         inline CommutationDO *setUpInitialCommutationDependableObject( BaseDependency const &target, AccessType const &accessType, TrackableObject &status );
         
         /*! \brief Adds a commutative access of a DependableObject to the domains dependency system.
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectCommutativeDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an inout access of a DependableObject to the domains dependency system.
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectInoutDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an output access of a DependableObject to the domains dependency system.
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] status status of the address/region
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectOutputDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an input region access of a DependableObject to the domains dependency system. 
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectInputDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
         
         /*! \brief Adds an output region without write access of a DependableObject to the domains dependency system. 
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectOutputNoWriteDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
          
         /*! \brief Adds a region without read access of a DependableObject to the domains dependency system. 
          *  \param depObj target DependableObject
          *  \param target accessed base address/region
          *  \param accessType kind of access
          *  \param[in,out] sourceStatus status of the source address/region (used to find input dependencies)
          *  \param[in,out] targetStatus status of the target address/region (used to represent the new access)
          *  \param callback Function to call if an immediate predecessor is found.
          */
         inline void submitDependableObjectInputNoReadDataAccess( DependableObject &depObj, BaseDependency const &target, AccessType const &accessType, TrackableObject &status, SchedulePolicySuccessorFunctor* callback );
         
         
      public:
         BaseDependenciesDomain ( ) :  DependenciesDomain(), _lastDepObjId ( 0 ) {
            NANOS_INSTRUMENT ( InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
            NANOS_INSTRUMENT ( _insKeyDeps[0] = ID->getEventKey("dependence"); )
            NANOS_INSTRUMENT ( _insKeyDeps[1] = ID->getEventKey("dep-direction"); )
            NANOS_INSTRUMENT ( _insKeyDeps[2] = ID->getEventKey("dep-address"); )
         }
         
         BaseDependenciesDomain ( const BaseDependenciesDomain &depDomain ) : DependenciesDomain( depDomain ), _lastDepObjId ( depDomain._lastDepObjId ) {
            NANOS_INSTRUMENT ( InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
            NANOS_INSTRUMENT ( _insKeyDeps[0] = ID->getEventKey("dependence"); )
            NANOS_INSTRUMENT ( _insKeyDeps[1] = ID->getEventKey("dep-direction"); )
            NANOS_INSTRUMENT ( _insKeyDeps[2] = ID->getEventKey("dep-address"); )
         }
   };

} // namespace nanos

#endif

