#include "system.hpp"
#include <cstdlib>

using namespace nanos;

namespace nanos {
namespace OpenMP {

int * ssCompatibility __attribute__((weak));

class OpenMPInit : public System::Init {
   public:
     void operator() () {
       if ( ssCompatibility != NULL) {
         sys.setInitialMode(System::POOL);
       } else {
         sys.setInitialMode(System::ONE_THREAD);
       }
     }
};

static OpenMPInit init;

}
}

namespace nanos {
  System::Init *externInit = &OpenMP::init;
}

