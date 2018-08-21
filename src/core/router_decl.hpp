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

#ifndef NANOS_ROUTER_DECL_HPP
#define NANOS_ROUTER_DECL_HPP

#include <set>
#include <vector>
#include "nanos-int.h"

namespace nanos {

class Router {

   private:
      memory_space_id_t _lastSource;
      std::vector<unsigned int> _memSpaces;

      Router( Router const &r );
      Router &operator=( Router const &r );
      
   public:
      Router();
      ~Router();
      void initialize();
      memory_space_id_t getSource( memory_space_id_t destination,
            std::set<memory_space_id_t> const &locs );
};

} // namespace nanos

#endif /* NANOS_ROUTER_DECL_HPP */
