#include "schedule.hpp"
#include "wddeque.hpp"
//#include "wf_sched.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

using namespace nanos;

typedef enum { FIFO, LIFO } wfPolicies;

static bool noStealParent = false;
static wfPolicies localPolicy = FIFO;
static wfPolicies stealPolicy = FIFO;


class WFData : public SchedulingData
{

      friend class WFPolicy; //in this way, the policy can access the readyQueue

   protected:
      WDDeque readyQueue;

   public:
      // constructor
      WFData( int id=0 ) : SchedulingData( id ) {}

      //TODO: copy & assigment costructor

      // destructor
      ~WFData() {}
};

class WFPolicy : public SchedulingGroup
{

   private:
      //policy variables for local dequeue and stealing: FIFO or LIFO?
      // int localPolicy;
      //int stealPolicy;
      //bool stealParent;

   public:
      // constructor
      WFPolicy() : SchedulingGroup( "wf-steal-sch" ) {} //, localPolicy(LIFO), stealPolicy(FIFO) {} //stealParent(true) {}

      WFPolicy( int groupsize ) : SchedulingGroup( "wf-steal-sch", groupsize ) {} //, localPolicy(FIFO), stealPolicy(FIFO) {}

      WFPolicy( int groupsize, int localP ) : SchedulingGroup( "wf-steal-sch", groupsize ) {} //, localPolicy(localP), stealPolicy(FIFO) {}

      WFPolicy( int groupsize, int localP, int stealPol ) : SchedulingGroup( "wf-steal-sch", groupsize ) {} //, localPolicy(localP), stealPolicy(stealPol) {}

      WFPolicy( int groupsize, int localP, int stealPol, bool stealPar ) : SchedulingGroup( "wf-steal-sch", groupsize ) {} //, localPolicy(localP), stealPolicy(stealPol) {}

      // TODO: copy and assigment operations
      // destructor
      virtual ~WFPolicy() {}

      virtual WD *atCreation ( BaseThread *thread, WD &newWD );
      virtual WD *atIdle ( BaseThread *thread );
      virtual void queue ( BaseThread *thread, WD &wd );
      virtual SchedulingData * createMemberData ( BaseThread &thread );
};

void WFPolicy::queue ( BaseThread *thread, WD &wd )
{
   WFData *data = ( WFData * ) thread->getSchedulingData();
   data->readyQueue.push_front( &wd );
}

WD * WFPolicy::atCreation ( BaseThread *thread, WD &newWD )
{
   //NEW: now it does not enqueue the created task, but it moves down to the generated son: DEPTH-FIRST
   return &newWD;
}

WD * WFPolicy::atIdle ( BaseThread *thread )
{

//     std::cout << "Dumping configuration:" << std::endl;
//        if(noStealParent == true) std::cout << "Not Stealing Parent! " << std::endl;
//        else std::cout << "Stealing Parent " << std::endl;

   WorkDescriptor * wd = NULL;

   WFData *data = ( WFData * ) thread->getSchedulingData();

   if ( ( ( localPolicy == LIFO ) && ( ( ( wd = data->readyQueue.pop_front( thread ) ) ) != NULL ) ) ||
         ( ( localPolicy == FIFO ) && ( ( wd = data->readyQueue.pop_back( thread ) ) != NULL ) ) ) {
      return wd;
   } else {
      if ( noStealParent == false ) {
         if ( ( wd = ( thread->getCurrentWD() )->getParent() ) != NULL ) {
            //removing it from the queue. Try to remove from one queue: if someone move it, I stop looking for it to avoid ping-pongs.
            if ( ( wd->isEnqueued() ) == true && ( !( wd )->isTied() || ( wd )->isTiedTo() == thread ) ) { //not in queue = in execution, in queue = not in execution
               if ( wd->getMyQueue()->removeWD( wd ) == true ) { //found it!
                  return wd;
               }
            }
         }
      }

      //next: steal from other queues
      int newposition = ( ( data->getSchId() ) +1 ) % getSize();

      //should be random: for now it checks neighbour queues in round robin
      while ( ( newposition != data->getSchId() ) && (
                 ( ( stealPolicy == LIFO ) && ( ( ( wd = ( ( ( WFData * ) ( getMemberData( newposition ) ) )->readyQueue.pop_back( thread ) ) ) == NULL ) ) ) ||
                 ( ( stealPolicy == FIFO ) && ( ( ( wd = ( ( ( WFData * ) ( getMemberData( newposition ) ) )->readyQueue.pop_front( thread ) ) ) == NULL ) ) )
              ) ) {
         newposition = ( newposition +1 ) % getSize();
      }

      return wd;
   }
}

SchedulingData * WFPolicy::createMemberData ( BaseThread &thread )
{
   return new WFData();
}


// Factories
SchedulingGroup * createWFPolicy ()
{
   return new WFPolicy();
}

SchedulingGroup * createWFPolicy ( int groupsize )
{
   return new WFPolicy( groupsize );
}


SchedulingGroup * createWFPolicy ( int localPolicy, int stealPolicy )
{
   return new WFPolicy( localPolicy, stealPolicy );
}

SchedulingGroup * createWFPolicy ( int groupsize, int localPolicy, int stealPolicy )
{
   return new WFPolicy( groupsize, localPolicy, stealPolicy );
}


SchedulingGroup * createWFPolicy ( int groupsize, int localPolicy, int stealPolicy, bool stealParent )
{
   return new WFPolicy( groupsize, localPolicy, stealPolicy, stealParent );
}


class WFSchedPlugin : public Plugin
{

   public:
      WFSchedPlugin() : Plugin( "WF scheduling Plugin",1 ) {}

      virtual void init() {
         Config config;

         //BUG: If defining local policy or steal policy the command line option *must not* include the = between the option name and the value, but a space
         config.registerArgOption( new Config::FlagOption( "nth-wf-no-steal-parent", noStealParent ) );

         Config::MapVar<wfPolicies>::MapList opts( 2 );
         opts[0] = Config::MapVar<wfPolicies>::MapOption( "FIFO", FIFO );
         opts[1] = Config::MapVar<wfPolicies>::MapOption( "LIFO", LIFO );
         config.registerArgOption( new Config::MapVar<wfPolicies>( "nth-wf-local-policy", localPolicy, opts ) );
         config.registerArgOption( new Config::MapVar<wfPolicies>( "nth-wf-steal-policy", stealPolicy, opts ) );

         config.init();

         std::cout << "localPolicy = " << localPolicy << ", stealPolicy = " << stealPolicy << std::endl;

         sys.setDefaultSGFactory( createWFPolicy );
      }
};

WFSchedPlugin NanosXPlugin;


