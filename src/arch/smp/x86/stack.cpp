#include "smpprocessor.hpp"

using namespace nanos;

void SMPDD::initStackDep ( void *userfuction, void *data, void *cleanup )
{
   state = stack;
   state += stackSize;
   *state = ( intptr_t )this;
   state--;
   *state = ( intptr_t )data;
   state--;
   *state = ( intptr_t )cleanup;
   state--;
   *state = ( intptr_t )userfuction;
   state--;

   // skip first state
   state -= 3;
}
