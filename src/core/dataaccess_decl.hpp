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

#ifndef _NANOS_DATA_ACCESS_DECL
#define _NANOS_DATA_ACCESS_DECL

#include "nanos-int.h"
#include <iosfwd>
// 
namespace nanos
{

  /*! \class DataAccess
   *  \brief Contains information about dependency data access necessary to add a \a DependableObject to the Dependencies system
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
          *  \param dimensionCount Number of dimensions
          *  \param dimensions Array of dimension descriptors from least significant to most significant
          */
         DataAccess ( void * addr, bool input, bool output,
                      bool canRename, bool commutative, short dimensionCount,
                      nanos_region_dimension_internal_t const *dimensions );

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

        /*! \brief returns true if there is commutativity over this access
         */
         bool isCommutative() const;

        /*! \brief sets the access to be commutative
         */
         void setCommutative( bool b );
         
   };
   
   
   namespace dependencies_domain_internal {
      class AccessType: public nanos_access_type_internal_t {
      public:
         AccessType()
            {
               input = 0;
               output = 0;
               can_rename = 0;
               commutative = 0;
            }
         
         AccessType(nanos_access_type_internal_t const &accessType)
            {
               input = accessType.input;
               output = accessType.output;
               can_rename = accessType.can_rename;
               commutative = accessType.commutative;
            }
         
         AccessType const &operator|=(nanos_access_type_internal_t const &accessType)
            {
               input |= accessType.input;
               output |= accessType.output;
               can_rename &= accessType.can_rename;
               commutative &= accessType.commutative;
               
               return *this;
            }
         friend std::ostream &operator<<( std::ostream &o, AccessType const &accessType);
      };
   } // namespace dependencies_domain_internal
}

#endif
