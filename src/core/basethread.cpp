#include "basethread.hpp"
#include "processingelement.hpp"
#include "system.hpp"

using namespace nanos;

__thread BaseThread * nanos::myThread=0;

Atomic<int> BaseThread::idSeed = 0;

void BaseThread::run ()
{
   associate();
   run_dependent();
}

void BaseThread::associate ()
{
   started = true;
   myThread = this;

   if ( sys.getBinding() ) bind();

   threadWD.tieTo( *this );

   setCurrentWD( threadWD );
}

bool BaseThread::singleGuard ()
{
   // return getTeam()->singleGuard(++local_single); # doesn't work
   // probably by some gcc bug
   local_single++;
   return getTeam()->singleGuard( local_single );
}

