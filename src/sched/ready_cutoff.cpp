#include "system.hpp"
#include "cutoff.hpp"
#include "plugin.hpp"


#define DEFAULT_CUTOFF_READY 100

using namespace nanos;


class ready_cutoff: public cutoff {
private:
    int max_ready;

public:
    ready_cutoff() : max_ready(DEFAULT_CUTOFF_READY) {}

    void init() {}
    void setMaxCutoff(int mr) { max_ready = mr; } 
    bool cutoff_pred();

    ~ready_cutoff() {}
};


bool ready_cutoff::cutoff_pred() {
   //checking if the number of ready tasks is higher than the allowed maximum
   if( sys.getReadyNum() > max_ready )  {
      verbose0("Cutoff Policy: avoiding task creation!");
      return false;
   }
   return true;
}

//factory
cutoff * createReadyCutoff() {
    return new ready_cutoff();
}


class ReadyCutOffPlugin : public Plugin
{
   public:
      ReadyCutOffPlugin() : Plugin("Ready Task CutOff Plugin",1) {}
      virtual void init() {
            sys.setCutOffPolicy(createReadyCutoff());
      }
};

ReadyCutOffPlugin NanosXPlugin;
