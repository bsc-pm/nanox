#ifndef REGIONSET_DECL_HPP
#define REGIONSET_DECL_HPP

#include <map>
#include "atomic_decl.hpp"
#include "globalregt_decl.hpp"
#include "newregiondirectory_decl.hpp"

namespace nanos {

class RegionSet {
   typedef std::map< reg_t, unsigned int > reg_set_t;
   typedef std::map< NewNewRegionDirectory::RegionDirectoryKey, reg_set_t > object_set_t;
   Lock         _lock;
   object_set_t _set;
   public:
   RegionSet();
   void addRegion( global_reg_t const &reg, unsigned int version );
   unsigned int hasRegion( global_reg_t const &reg );
   bool hasObjectOfRegion( global_reg_t const &reg );

   bool hasVersionInfoForRegion( global_reg_t const &reg, unsigned int &version, NewLocationInfoList &locations );
};

}
#endif /* REGIONSET_DECL_HPP */
