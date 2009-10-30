#include "system.hpp"
#include "cutoff.hpp"
#include "plugin.hpp"


#define DEFAULT_CUTOFF_NUM 1000

using namespace nanos;


class tasknum_cutoff: public cutoff
{

   private:
      int max_cutoff;

   public:
      tasknum_cutoff() : max_cutoff( DEFAULT_CUTOFF_NUM ) {}

      void init() {}

      void setMaxCutoff( int mc ) { max_cutoff = mc; }

      bool cutoff_pred();

      ~tasknum_cutoff() {}
};


bool tasknum_cutoff::cutoff_pred()
{
   if ( sys.getTaskNum() > max_cutoff ) {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
cutoff * createTasknumCutoff()
{
   return new tasknum_cutoff();
}



class TasknumCutOffPlugin : public Plugin
{

   public:
      TasknumCutOffPlugin() : Plugin( "TaskNum CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createTasknumCutoff() );
      }
};

TasknumCutOffPlugin NanosXPlugin;
