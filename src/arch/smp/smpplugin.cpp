#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "smpdd.hpp"
#include "system.hpp"

using namespace nanos;

PE * smpProcessorFactory ( int id )
{
   return new SMPProcessor( id );
}

class SMPPlugin : public Plugin
{

   public:
      SMPPlugin() : Plugin( "SMP PE Plugin",1 ) {}

      virtual void init() {
         sys.setHostFactory( smpProcessorFactory );

         Config config;
         SMPProcessor::prepareConfig( config );
         SMPDD::prepareConfig( config );
         config.init();
      }
};

SMPPlugin NanosXPlugin;



