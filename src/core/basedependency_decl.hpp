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

#ifndef _NANOS_BASE_DEPENDENCY_DECL_H
#define _NANOS_BASE_DEPENDENCY_DECL_H

namespace nanos {

   //! \class Base dependency class.
   //! \brief Abstract class to represent a memory address dependency or a region.
   class BaseDependency
   {
      public:
         //! \brief Base dependency default constructor.
         BaseDependency () {}

         //! \brief Base dependency copy constructor
         //! \param obj another base dependency
         BaseDependency ( const BaseDependency &obj ) {}

         //! \brief Base dependency destructor
         virtual ~BaseDependency () {}
         
         //! \brief Clones the dependency object.
         virtual BaseDependency* clone() const = 0;

         //! \brief Returns dependency base address
         virtual void * getAddress () const = 0;
                  
        /*! \brief Check if two dependencies overlap/collide.
         */
         virtual bool overlap ( const BaseDependency &obj ) const = 0;
   };

} // namespace nanos

#endif
