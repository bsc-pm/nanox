#include "system.hpp"
#include "cutoff.hpp"
#include "plugin.hpp"

#define DEFAULT_CUTOFF_IDLE 5

using namespace nanos;

//TODO: only works with 1 scheduling group

class idle_cutoff: public cutoff {
private:
    int max_idle;

public:
    idle_cutoff() : max_idle(DEFAULT_CUTOFF_IDLE) {}

    void init() {}
    void setMaxCutoff(int mi) { max_idle = mi; } 
    bool cutoff_pred();

    ~idle_cutoff() {}
};


bool idle_cutoff::cutoff_pred() {
   //checking if the number of idle tasks is higher than the allowed maximum
   if( sys.getIdleNum() > max_idle )  {
      verbose0("Cutoff Policy: avoiding task creation!");
      return false;
   }
   return true;
}

//factory
cutoff * createIdleCutoff() {
    return new idle_cutoff();
}



class IdleCutOffPlugin : public Plugin
{
   public:
      IdleCutOffPlugin() : Plugin("Idle Threads CutOff Plugin",1) {}
      virtual void init() {
           sys.setCutOffPolicy(createIdleCutoff());
      }
};

IdleCutOffPlugin NanosXPlugin;
