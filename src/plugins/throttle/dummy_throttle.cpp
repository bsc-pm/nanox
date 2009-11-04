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

#include "throttle.hpp"

namespace nanos {
namespace ext {

class DummyThrottle: public ThrottlePolicy
{

   private:
      /*!
        we decide once if all new tasks are to be created during the execution
        if _createTasks is true, then we have the maximum number of tasks else we have only one task (sequential comp.)
      */
      bool _createTasks;
      static const bool _default;

   public:
      DummyThrottle() : _createTasks( _default ) {}

      void setCreateTask( bool ct ) { _createTasks = ct; }

      bool throttle();

      ~DummyThrottle() {};
};

const bool DummyThrottle::_default = false;

bool DummyThrottle::throttle()
{
   return _createTasks;
}

//factory
DummyThrottle * createDummyThrottle()
{
   return new DummyThrottle();
}

}}
