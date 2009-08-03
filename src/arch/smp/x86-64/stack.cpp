#include "smpprocessor.hpp"

using namespace nanos;

extern "C" {
// low-level helper routine to start a new user-thread
void startHelper ();
}

void SMPWD::initStackDep ( void *userfuction, void *cleanup )
{
   state = stack;
   state += stackSize;
   
   *state = (intptr_t)cleanup; state--;
   *state = (intptr_t)this; state--;
   *state = (intptr_t)userfuction; state --;
   *state = (intptr_t)getData(); state--;
   *state = (intptr_t)startHelper; state--;   
   // skip first state
   state -= 5; 
}
