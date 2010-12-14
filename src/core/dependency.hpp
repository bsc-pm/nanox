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

#ifndef _NANOS_DEPENDENCY
#define _NANOS_DEPENDENCY

#include "nanos-int.h"

namespace nanos
{

  /*! \class Dependency
   *  \brief Contains information about dependencies necessary to add a DependableObject to the Dependencies system
   */
   class Dependency : public nanos_dependence_internal_t
   {
      public:
         /*! \brief Dependency default constructor
          *
          *  \param address Address of the dependency's address 
          *  \param input Whether the dependency is input or not 
          *  \param output Whether the dependency is output or not
          *  \param canRename Whether the dependency can rename or not
          */
         Dependency ( void ** addr = NULL, ptrdiff_t offset = 0, bool input = false, bool output = false,
                      bool canRename = false, bool commutative = false, size_t storageSize = 0 )
         {
            address = addr;
            offset = offset;
            flags.input = input;
            flags.output = output;
            flags.can_rename = canRename;
            flags.commutative = commutative;
            size = storageSize;
         }
         /*! \brief Dependency copy constructor
          *
          *  \param obj another Dependency
          */
         Dependency ( const Dependency &dep )
         {
            address = dep.address;
            offset = dep.offset;
            flags.input = dep.flags.input;
            flags.output = dep.flags.output;
            flags.can_rename = dep.flags.can_rename;
            flags.commutative = dep.flags.commutative;
            size = dep.size;
         }
        /*! \brief Dependency copy asssignment operator, can be self-assigned.
         *
         *  \param obj another Dependency
         */
         const Dependency & operator= ( const Dependency &dep )
         {
            if ( this == &dep ) return *this; 
            address = dep.address;
            offset = dep.offset;
            flags.input = dep.flags.input;
            flags.output = dep.flags.output;
            flags.can_rename = dep.flags.can_rename;
            flags.commutative = dep.flags.commutative;
            size = dep.size;
            return *this;
         }
         /*! \brief Dependency destructor
          */
         ~Dependency () {}
         
        /*! \brief Obtain the dependency's address address
         */
         void ** getAddress() const
         { return address; }
         
        /*! \brief Obtain the dependency's address address
         */
         ptrdiff_t getOffset() const
         { return offset; }

        /*! \brief Compute the dependency address
         */
         void * getDepAddress() const
         { return (void *) ( (char *) (*address) + offset ); }
         
        /*! \brief returns true if it is an input dependency
         */
         bool isInput() const
         { return flags.input; }

        /*! \brief sets the dependency input clause to b
         */
         void setInput( bool b )
         { flags.input = b; }
         
        /*! \brief returns true if it is an output dependency
         */
         bool isOutput() const
         { return flags.output; }

        /*! \brief sets the dependency output clause to b
         */
         void setOutput( bool b )
         { flags.output = b;}
         
        /*! \brief return true if the dependency can rename
         */
         bool canRename() const
         { return flags.can_rename; }

        /*! \brief sets the rename attribute to b
         */
         void setCanRename( bool b )
         { flags.can_rename = b; }

        /*! \brief returns true if there is a commutative over this dependency
         */
         bool isCommutative() const
         { return flags.commutative; }

        /*! \brief sets the dependency to be a commutative
         */
         void setCommutative( bool b )
         { flags.commutative = b;}
         
   };
}

#endif
