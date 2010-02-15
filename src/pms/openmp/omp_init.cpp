#include "system.hpp"
#include <cstdlib>

using namespace nanos;

namespace nanos {
namespace OpenMP {

  int * ssCompatibility __attribute__((weak));

  static void ompInit()
  {
     if ( ssCompatibility != NULL) {
       sys.setInitialMode(System::POOL);
     } else {
       sys.setInitialMode(System::ONE_THREAD);
     }
  }

}
}

namespace nanos {
  System::Init externInit = OpenMP::ompInit;
}

