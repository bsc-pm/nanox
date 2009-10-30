#include "system.hpp"
#include "cutoff.hpp"
#include "plugin.hpp"


#define DEFAULT_CUTOFF_LEVEL 2

using namespace nanos;


class level_cutoff: public cutoff
{

   private:
      int max_level;

   public:
      level_cutoff() : max_level( DEFAULT_CUTOFF_LEVEL ) {}

      void init() {}

      void setMaxCutoff( int ml ) { max_level = ml; }

      bool cutoff_pred();

      ~level_cutoff() {}
};


bool level_cutoff::cutoff_pred()
{
   //checking the parent level of the next work to be created (check >)
   if ( ( myThread->getCurrentWD() )->getLevel() > max_level )  {
      verbose0( "Cutoff Policy: avoiding task creation!" );
      return false;
   }

   return true;
}

//factory
cutoff * createLevelCutoff()
{
   return new level_cutoff();
}


class LevelCutOffPlugin : public Plugin
{

   public:
      LevelCutOffPlugin() : Plugin( "Task Tree Level CutOff Plugin",1 ) {}

      virtual void init() {
         sys.setCutOffPolicy( createLevelCutoff() );
      }
};

LevelCutOffPlugin NanosXPlugin;
