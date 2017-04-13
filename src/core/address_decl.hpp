/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
         size_t                  _size; /**< Size in bytes of the dependency object */
      public:

        /*! \brief Address default constructor
         *  Creates an Address with the given address associated.
         */
         Address ( TargetType address = NULL, size_t s = 1 )
            : _address( address ), _size( s ) {}

        /*! \brief Address copy constructor
         *  \param obj another Address
         */
         Address ( const Address &obj ) 
            :  BaseDependency(), _address ( obj._address ), _size( obj._size ) {}

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

         //! \brief Returns the size of the dependency object.
         virtual size_t size() const;

         //! \brief Returns the size of the dependency object.
         virtual void size ( size_t s );
         
        /*! \brief Overlap operator.
         */
         virtual bool overlap ( const BaseDependency &obj ) const;
   };

} // namespace nanos

#endif
