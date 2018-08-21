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

#ifndef _NANOS_DATA_ACCESS_DECL
#define _NANOS_DATA_ACCESS_DECL

#include "nanos-int.h"
#include <cstddef>
#include <iosfwd>
// 
namespace nanos {

  /*! \class DataAccess
   * 
   *  \brief Contains information about dependency data access necessary to add a \a DependableObject to the Dependencies system
   *
   *  The DataAccess class encapsulates the necessary information to identify and classify dependencies.
   *
   *  - address: Reference to the address used to identify a dependency.
   *  - input: Determines if the storage pointed by address is read by the user of the dependency.
   *  - output: Determines if the storage pointed by address is written by the user of the dependency.
   *  - canRename: Currently unused. Intended to allow renaming or not for the dependency.
   *  - size: Currently unused. Size of the storage pointed by address. Intended to use when renaming is allowed.
   */
   class DataAccess : public nanos_data_access_internal_t
   {
      public:
         /*! \brief DataAccess default constructor
          *
          *  \param address Address of the data 
          *  \param input Whether the access is input or not 
          *  \param output Whether the access is output or not
          *  \param canRename Whether the access can rename or not
          *  \param concurrent If the accesses can run concurrently
          *  \param commutative If accesses may occur in any order, but not concurrently
          *  \param dimensionCount Number of dimensions
          *  \param dimensions Array of dimension descriptors from least significant to most significant
          *  \param offset Offset of the first element
          */
         DataAccess ( void * addr, bool input, bool output,
                      bool canRename, bool concurrent, bool commutative,
                      short dimensionCount, nanos_region_dimension_internal_t const *dimensions,
                      ptrdiff_t offset );

         /*! \brief DataAccess copy constructor
          *
          *  \param dataAccess another DataAccess
          */
         DataAccess ( const DataAccess &dataAccess );

        /*! \brief DataAccess copy asssignment operator, can be self-assigned.
         *
         *  \param dataAccess another DataAccess
         */
         const DataAccess & operator= ( const DataAccess &dataAccess );

         /*! \brief DataAccess destructor
          */
         ~DataAccess () {}
         
        /*! \brief Obtain the base address of the access
         */
         void * getAddress() const;
         
        /*! \brief Obtain the dependency's address address
    */
    ptrdiff_t getOffset() const;
         
        /*! \brief Compute the address of the first element
         * (address + offset)
         */
         void * getDepAddress() const;
         
        /*! \brief returns true if it is an input access
         */
         bool isInput() const;

        /*! \brief sets the access input clause to b
         */
         void setInput( bool b );
         
        /*! \brief returns true if it is an output access
         */
         bool isOutput() const;

        /*! \brief sets the access output clause to b
         */
         void setOutput( bool b );
         
        /*! \brief return true if the access can rename
         */
         bool canRename() const;

        /*! \brief sets the access attribute to b
         */
         void setCanRename( bool b );

        /*! \brief returns true if this access can occur concurrent with other concurrent accesses
         */
         bool isConcurrent() const;

        /*! \brief sets the access to be concurrent
         */
         void setConcurrent( bool b );

        /*! \brief returns true if this access can be reordered with other commutative accesses
          */
         bool isCommutative() const;

        /*! \brief sets the access to be commutative
         */
         void setCommutative( bool b );
         
        /*! \brief gets the size of the dataAccess element
          */
        std::size_t getSize() const;        
        
        /*! \brief gets the pointer of the dimensions
          */
        nanos_region_dimension_internal_t const* getDimensions() const;
        
        /*! \brief gets the number of dimensions
          */
        short getNumDimensions() const;
   };
   
   
   namespace dependencies_domain_internal {
      class AccessType: public nanos_access_type_internal_t {
      public:
         AccessType()
            {
               input = 0;
               output = 0;
               // can_rename: see below remark.
               can_rename = true;
               /*
                * #605 (gmiranda): this has to be initialised to true, otherwise
                * operator&= will always set it to false.
                * When we have 2 accesses to the same region, if one of them is
                * concurrent and the other is not, we want it to be non
                * concurrent.
                */
               concurrent = true; 
               commutative = true;
            }
         
         AccessType(nanos_access_type_internal_t const &accessType)
            {
               input = accessType.input;
               output = accessType.output;
               can_rename = accessType.can_rename;
               concurrent = accessType.concurrent;
               commutative = accessType.commutative;
            }
         
         AccessType const &operator|=(nanos_access_type_internal_t const &accessType)
            {
               input |= accessType.input;
               output |= accessType.output;
               can_rename &= accessType.can_rename;
               concurrent &= accessType.concurrent;
               commutative &= accessType.commutative;
               
               return *this;
            }
         friend std::ostream &operator<<( std::ostream &o, AccessType const &accessType);
      };
   } // namespace dependencies_domain_internal

} // namespace nanos

#endif
