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

#ifndef _NANOS_COPYDATA_DECL
#define _NANOS_COPYDATA_DECL

#include <iostream>
#include "nanos-int.h"

namespace nanos {

  /*! \class CopyData
   *  \brief Contains information about Copies
   */
   class CopyData : public nanos_copy_data_internal_t
   {
      public:
        /*! \brief CopyData default constructor
         *  \param address Address of the CopyData's address 
         *  \param input Whether the CopyData is input or not 
         *  \param output Whether the CopyData is output or not
         */
         CopyData ( uint64_t addr = ( uint64_t ) NULL, nanos_sharing_t nxSharing = NANOS_SHARED, bool input = false,
                    bool output = false, std::size_t numDimensions = 0, nanos_region_dimension_internal_t const *dims = NULL, ptrdiff_t off = 0, uint64_t hostBaseAddress = 0, reg_t hostRegionId = 0 );

        /*! \brief CopyData copy constructor
         *  \param obj another CopyData
         */
         CopyData ( const CopyData &cd );

        /*! \brief CopyData copy assignment operator, can be self-assigned.
         *  \param obj another CopyData
         */
         const CopyData & operator= ( const CopyData &cd );

        /*! \brief CopyData destructor
         */
         ~CopyData () {}
         
        /*! \brief Obtain the CopyData's address address
         */
         void *getBaseAddress() const;
         
        /*! \brief Set the CopyData's address address
         */
         void setBaseAddress( void *addr );
         
        /*! \brief returns true if it is an input CopyData
         */
         bool isInput() const;

        /*! \brief sets the CopyData input clause to b
         */
         void setInput( bool b );
         
        /*! \brief returns true if it is an output CopyData
         */
         bool isOutput() const;

        /*! \brief sets the CopyData output clause to b
         */
         void setOutput( bool b );
         
        /*! \brief  returns the CopyData's size
         */
         std::size_t getSize() const;
         std::size_t getMaxSize() const;
         std::size_t getFitSize() const;
         uint64_t getFitAddress() const;

        /*! \brief Returns true if the data to copy is shared
         */
         bool isShared() const;

        /*! \brief Returns true if the data to copy is private
         */
         bool isPrivate() const;

         nanos_sharing_t getSharing() const;

         std::size_t getNumDimensions() const;
         void setNumDimensions( std::size_t ndims );
         nanos_region_dimension_internal_t const *getDimensions() const;
         void setDimensions(nanos_region_dimension_internal_t *);
         
         uint64_t getAddress() const ;
         uint64_t getOffset() const ;
         uint64_t getHostBaseAddress() const ;
         void setHostBaseAddress(uint64_t addr);
         void getFitDimensions( nanos_region_dimension_internal_t *outDimensions ) const;
         void setHostRegionId( reg_t id );
         reg_t getHostRegionId() const;
         bool isRemoteHost() const;
         void setRemoteHost( bool value );
         void deductCd( CopyData const &ref, nanos_region_dimension_internal_t *newDims ) const;
         bool equalGeometry( CopyData const &cd ) const;
         void setDeductedCD( CopyData *cd );
         CopyData *getDeductedCD();

      friend std::ostream& operator<< (std::ostream& o, CopyData const &cd);

      private:
         size_t getFitSizeRecursive( int i ) const;
         size_t getWideSizeRecursive( int i ) const;
         uint64_t getFitOffsetRecursive( int i ) const;

   };
   std::ostream& operator<< (std::ostream& o, CopyData const &cd);

} // namespace nanos

#endif
