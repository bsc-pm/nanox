#include "workgroup.hpp"
#include "atomic.hpp"
#include "schedule.hpp"

using namespace nanos;

Atomic<int> WorkGroup::atomicSeed( 0 );

void WorkGroup::addWork ( WorkGroup &work )
{
   components++;
   work.addToGroup( *this );
}

void WorkGroup::addToGroup ( WorkGroup &parent )
{
   partOf.push_back( &parent );
}

void WorkGroup::exitWork ( WorkGroup &work )
{
   components--;
}

void WorkGroup::sync ()
{
   phase_counter++;
   //TODO: block and switch

   while ( phase_counter < components );

   //TODO: reinit phase_counter
}

void WorkGroup::waitCompletation ()
{
   Scheduler::blockOnCondition<int>( &components.override(),0 );
}

void WorkGroup::done ()
{
   for ( ListOfWGs::iterator it = partOf.begin();
         it != partOf.end();
         it++ ) {
      ( *it )->exitWork( *this );
      //partOf.erase(it);
   }
}

WorkGroup::~WorkGroup ()
{
   done();
}
