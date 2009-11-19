#include "smpprocessor.hpp"

using namespace nanos::ext;

extern "C" {
// low-level helper routine to start a new user-thread
void startHelper ();
}


/*! \brief initializes a stack state for PowerPC32
 *
 * -------------------------------------------------------------------
 * Initial STACK state for PPC 32
 * -------------------------------------------------------------------
 *
 *  +------------------------------+
 *  |                              |
 *  +------------------------------+ <- state
 *  |    scratch area for helper   |
 *  +------------------------------+
 *  |     (20)  r12                |
 *  |     (24)  r14 =  userf       |
 *  |     (28)  r15 =  data        |
 *  |     (32)  r16 =  cleanup     |
 *  |     (36)  r17 =  clenaup arg |
 *  |           ....               |
 *  |     (92)  r31                |
 *  |     (96)  f14   (8 bytes)    |
 *  |           ....               |
 *  |    (232)  f31   (8 bytes)    |
 *  |    (240)  back chain = 0     |
 *  |    (244)  r0  =  startHelper |
 *  +------------------------------+
 *
 * \sa switchStacks
 */

void SMPDD::initStackDep ( void *userfunction, void *data, void *cleanup )
{
   // stack grows down
   _state = _stack;
   _state += _stackSize;

   _state -= 62; // 244/sizeof(intptr_t)

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
