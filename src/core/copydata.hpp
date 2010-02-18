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

#ifndef _NANOS_COPYDATA
#define _NANOS_COPYDATA

#include "nanos-int.h"

namespace nanos
{

  /*! \class CopyData
   *  \brief Contains information about dependencies necessary to add a DependableObject to the Dependencies system
   */
   class CopyData : public nanos_copy_data_internal_t
   {
      public:
         
        /*! \brief Constructor
         *  \param address Address of the dependency's address 
         *  \param input Whether the dependency is input or not 
         *  \param output Whether the dependency is output or not
         *  \param canRename Whether the dependency can rename or not
         */
         CopyData ( void * addr = NULL, bool input = false, bool output = false, size_t storageSize = 0 )
         {
            address = addr;
            flags.input = input;
            flags.output = output;
            size = storageSize;
         }

        /*! \brief Copy constructor
         *  \param obj another CopyData
         */
         CopyData ( const CopyData &cd )
         {
            address = cd.address;
            flags.input = cd.flags.input;
            flags.output = cd.flags.output;
            size = cd.size;
         }
 
        /*! \brief Destructor
         */
         ~CopyData () {}
         
        /*! \brief Assign operator, can be self-assigned.
         *  \param obj another CopyData
         */
         const CopyData & operator= ( const CopyData &cd )
         {
            if ( this == &cd ) return *this; 
            address = cd.address;
            flags.input = cd.flags.input;
            flags.output = cd.flags.output;
            size = cd.size;
            return *this;
         }
         
        /*! \brief Obtain the dependency's address address
         */
         void * getAddress() const
         { return address; }
         
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
         
        /*! \brief  returns the CopyData's size
         */
         size_t getSize() const
         { return size; } 
   };
}

#endif
