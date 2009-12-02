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
#include <stdlib.h>

namespace nanos
{

  /*! \class Dependency
   *  \brief Contains information about dependencies necessary to add a DependableObject to the Dependencies system
   */
   class Dependency
   {
      private:
         /**< Address of the dependency's address */
         void ** _address;
         /**< Whether the dependency is input or not */
         bool _input;
         /**< Whether the dependency is output or not */
         bool _output;
         /**< Whether the dependency can rename or not */
         bool _canRename;

      public:
         
        /*! \brief Constructor
         *  \param address Address of the dependency's address 
         *  \param input Whether the dependency is input or not 
         *  \param output Whether the dependency is output or not
         *  \param canRename Whether the dependency can rename or not
         */
         Dependency (void ** address = NULL, bool input = false, bool output = false, bool canRename = false) : _address ( address ), _input ( input ), _output ( output ), _canRename ( canRename ) {}

        /*! \brief Copy constructor
         *  \param obj another Dependency
         */
         Dependency ( const Dependency &dep ) :  _address ( dep._address ), _input(dep._input), _output(dep._output), _canRename(dep._canRename) {}
 
        /*! \brief Destructor
         */
         ~Dependency () {}
         
        /*! \brief Assign operator, can be self-assigned.
         *  \param obj another Dependency
         */
         const Dependency & operator= ( const Dependency &dep )
         {
            if ( this == &dep ) return *this; 
            _address = dep._address;
            _input = dep._input;
            _output = dep._output;
            _canRename = dep._canRename;
            return *this;
         }
         
        /*! \brief Obtain the dependency's address address
         */
         void ** getAddress() const
         { return _address; }
         
        /*! \brief returns true if it is an input dependency
         */
         bool isInput() const
         { return _input; }

        /*! \brief sets the dependency input clause to b
         */
         void setInput( bool b )
         { _input = b; }
         
        /*! \brief returns true if it is an output dependency
         */
         bool isOutput() const
         { return _output; }

        /*! \brief sets the dependency output clause to b
         */
         void setOutput( bool b )
         { _output = b;}
         
        /*! \brief return true if the dependency can rename
         */
         bool canRename() const
         { return _canRename; }

        /*! \brief sets the rename attribute to b
         */
         void setCanRename( bool b )
         { _canRename = b; }
   };
}

#endif
