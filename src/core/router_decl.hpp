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

}

#endif /* NANOS_ROUTER_DECL_HPP */
