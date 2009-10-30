#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

using namespace nanos;

class BreadthFirstPolicy : public SchedulingGroup
{

   private:
      WDDeque   readyQueue;
      bool      useStack;

   public:
      // constructor
      BreadthFirstPolicy( bool stack, int groupSize ) : SchedulingGroup( "breadth-first-sch",groupSize ), useStack( stack ) {}

      // TODO: copy and assigment operations
      // destructor
      virtual ~BreadthFirstPolicy() {}

      virtual WD *atCreation ( BaseThread *thread, WD &newWD );
      virtual WD *atIdle ( BaseThread *thread );
      virtual void queue ( BaseThread *thread, WD &wd );
};

void BreadthFirstPolicy::queue ( BaseThread *thread, WD &wd )
{
   readyQueue.push_back( &wd );
}

WD * BreadthFirstPolicy::atCreation ( BaseThread *thread, WD &newWD )
{
   queue( thread,newWD );
   return 0;
}

WD * BreadthFirstPolicy::atIdle ( BaseThread *thread )
{
   if ( useStack ) return readyQueue.pop_back( thread );

   return readyQueue.pop_front( thread );
}

static bool useStack = false;

// Factory
SchedulingGroup * createBreadthFirstPolicy ( int groupSize )
{
   return new BreadthFirstPolicy( useStack,groupSize );
}

class BFSchedPlugin : public Plugin
{

   public:
      BFSchedPlugin() : Plugin( "BF scheduling Plugin",1 ) {}

      virtual void init() {
         Config config;

         config.registerArgOption( new Config::FlagOption( "nth-bf-use-stack",useStack ) );
         config.registerArgOption( new Config::FlagOption( "nth-bf-stack",useStack ) );
         config.init();

         sys.setDefaultSGFactory( createBreadthFirstPolicy );
      }
};

BFSchedPlugin NanosXPlugin;

