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

#ifndef _NANOS_ADDRESS_DECL_H
#define _NANOS_ADDRESS_DECL_H

#include "basedependency_decl.hpp"
#include "trackableobject_decl.hpp"

namespace nanos {

  /*! \class Address
   *  \brief Represents a memory address
   */
   class Address : public BaseDependency
   {
      public:
         typedef void*           TargetType;
      private:
         TargetType              _address; /**< Pointer to the dependency address */
      public:

        /*! \brief Address default constructor
         *  Creates an Address with the given address associated.
         */
         Address ( TargetType address = NULL )
            : _address( address ) {}

        /*! \brief Address copy constructor
         *  \param obj another Address
         */
         Address ( const Address &obj ) 
            :  BaseDependency(), _address ( obj._address ) {}

        /*! \brief Address destructor
         */
         ~Address () {}

        /*! \brief Address assignment operator, can be self-assigned.
         *  \param obj another Address
         */
         const Address & operator= ( const Address &obj );
         
        /*! \brief Returns the address.
         */
         const TargetType& operator() () const;
         
        /*! \brief Clones the address.
         */
         BaseDependency* clone() const;
         
        /*! \brief Comparison operator.
         */
         bool operator== ( const Address &obj ) const;
         
         bool operator< ( const Address &obj ) const;

         //! \brief Returns dependence base address
         virtual void * getAddress () const;
         
        /*! \brief Overlap operator.
         */
         virtual bool overlap ( const BaseDependency &obj ) const;
   };

} // namespace nanos

#endif
