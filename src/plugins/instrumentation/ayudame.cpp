#define version_str "!"

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "Ayudame.h"
#include "smpdd.hpp" // FIXME: this the include should not be here (just testing smpdd)
namespace nanos {

class InstrumentationAyudame: public Instrumentation {
#ifndef NANOS_INSTRUMENTATION_ENABLED
public:
	// constructor
	InstrumentationAyudame() :
			Instrumentation() {
	}
	// destructor
	~InstrumentationAyudame() {
	}

	// low-level instrumentation interface (mandatory functions)
	void initialize(void) {
	}
	void finalize(void) {
	}
	void disable(void) {
	}
	void enable(void) {
	}
	void addResumeTask(WorkDescriptor &w) {
	}
	void addSuspendTask(WorkDescriptor &w, bool last) {
	}
	void addEventList(unsigned int count, Event *events) {
	}
	void threadStart(BaseThread &thread) {
	}
	void threadFinish(BaseThread &thread) {
	}
#else
public:
	// constructor
	InstrumentationAyudame() : Instrumentation( *NEW InstrumentationContextDisabled() ) {}
	// destructor
	~InstrumentationAyudame ( ) {}

	// low-level instrumentation interface (mandatory functions)
	void initialize( void )
	{
		printf( "###############################################\n"
				"################ INFORMATION ##################\n"
				"###############################################\n"
				"#                                             #\n"
				"#    THIS IS AN EMPTY INSTRUMENTATION FILE    #\n"
				"#         IF YOU WANT TO USE TEMANEJO         #\n"
				"#                PLEASE VISIT                 #\n"
				"#                                             #\n"
				"#             www.hlrs.de/temanejo‎            #\n"
				"#                                             #\n"
				"#      FOR DOWNLOADS AND MORE INFORMATION     #\n"
				"#                                             #\n"
				"###############################################\n"
				"###############################################\n");
	}
	void finalize( void ) {}
	void disable( void ) {}
	void enable( void ) {}
	void addResumeTask( WorkDescriptor &w ) {}
	void addSuspendTask( WorkDescriptor &w, bool last ) {}
	void addEventList ( unsigned int count, Event *events ) {}
	void threadStart( BaseThread &thread ) {}
	void threadFinish ( BaseThread &thread ) {}
#endif

};

namespace ext {

class InstrumentationAyudamePlugin: public Plugin {
public:
	InstrumentationAyudamePlugin() :
			Plugin(
					"Instrumentation which implements Ayudame/Temanejo protocol.",
					1) {
	}
	~InstrumentationAyudamePlugin() {
	}

	void config(Config &cfg) {
	}

	void init() {
		sys.setInstrumentation(new InstrumentationAyudame());
	}
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-ayudame",
		nanos::ext::InstrumentationAyudamePlugin);
