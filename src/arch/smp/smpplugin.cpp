#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace nanos;

PE * smpProcessorFactory (int id)
{
   return new SMPProcessor(id);
}

class SMPPlugin : public Plugin
{
   public:
      SMPPlugin() : Plugin("SMP PE Plugin",1) {}
      virtual void init() {
           sys.setHostFactory(smpProcessorFactory);
      }
};

SMPPlugin NanosXPlugin;



