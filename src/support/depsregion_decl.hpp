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

#ifndef _NANOS_DEPSREGION_DECL_H
#define _NANOS_DEPSREGION_DECL_H

#include "basedependency_decl.hpp"
#include "trackableobject_decl.hpp"

namespace nanos {

  /*! \class DepsRegion
   *  \brief Represents a memory address
   */
   class DepsRegion : public BaseDependency
   {
      public:
         typedef void*           TargetType;
      private:
         TargetType              _address; /**< Pointer to the dependency address */
         TargetType _endAddress;
         TrackableObject* _trackable; /** Trackable which represents this region */
         short _dimensionCount;
         std::vector<nanos_region_dimension_internal_t> _dimensions;
      public:

        /*! \brief DepsRegion default constructor
         *  Creates an DepsRegion with the given address associated.
         */
         DepsRegion ( TargetType address = NULL, TargetType endAddress = NULL, TrackableObject* trackable = NULL, short dimensionCount=0, const nanos_region_dimension_internal_t *dimensions = NULL )
            : _address( address ) , _endAddress( endAddress ), _trackable(trackable), _dimensionCount(dimensionCount), _dimensions() {
            if (_dimensionCount>0) {
                _dimensions.reserve(_dimensionCount);
                for (short i=0; i<dimensionCount; i++) {
                    _dimensions.push_back(dimensions[i]);
                }
            }
        }

        /*! \brief DepsRegion copy constructor
         *  \param obj another DepsRegion
         */
         DepsRegion ( const DepsRegion &obj ) 
            :  BaseDependency(), _address ( obj._address ), _endAddress ( obj._endAddress ),  _trackable( obj._trackable ),
            _dimensionCount( obj._dimensionCount ), _dimensions( obj._dimensions ) {}

        /*! \brief DepsRegion destructor
         */
         ~DepsRegion () {}

        /*! \brief DepsRegion assignment operator, can be self-assigned.
         *  \param obj another DepsRegion
         */
         const DepsRegion & operator= ( const DepsRegion &obj );
         
        /*! \brief Returns the address.
         */
         const TargetType& operator() () const;
         
        /*! \brief Clones the address.
         */
         BaseDependency* clone() const;
         
        /*! \brief Comparison operator.
         */
         bool operator== ( const DepsRegion &obj ) const;
         
        /*! \brief Overlap regions operator.
         */
         bool overlap ( const BaseDependency &obj ) const;
         
         bool operator< ( const DepsRegion &obj ) const;

         //! \brief Returns dependence base address
         virtual void * getAddress () const;
         
         //! \brief Returns size
         virtual void * getEndAddress() const;         
         
         //! \brief Returns size
         virtual size_t getSize() const;
         

         TrackableObject* getTrackable() const {
            return _trackable;
         }

         void setTrackable(TrackableObject* trackable) {
            _trackable = trackable;
         }
         
   };
} // namespace nanos

#endif
