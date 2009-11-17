#include "smpprocessor.hpp"

using namespace nanos::ext;

/* -------------------------------------------------------------------
 * Initial STACK state for PPC 32
 * -------------------------------------------------------------------
 *
 *  +------------------------------+
 *  |                              |
 *  +------------------------------+ <- state
 *  |
 *  | 
 *
 * -----------------------------------------------------------------*/

extern "C" {
// low-level helper routine to start a new user-thread
void startHelper ();
}

void SMPDD::initStackDep ( void *userfunction, void *data, void *cleanup )
{
   // stack grows down
   _state = _stack;
   _state += _stackSize;

   _state -= 60 + 2;

   // return link
   _state[61] = (intptr_t) startHelper;
   // back chain
   _state[60] = 0;
      
   // (r14) userf
   _state[6] = (intptr_t) userfunction;
   // (r15) data
   _state[7] = (intptr_t) data;
   // (r16) cleanup
   _state[8] = (intptr_t) cleanup;
   // (r17) cleanup arg
   _state[9] = (intptr_t) this;
}
