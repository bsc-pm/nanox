#include "plugin.hpp"
#include "slicer.hpp"
#include "system.hpp"

namespace nanos {
namespace ext {


class SlicerDynamicForPlugin : public Plugin {
   public:
	SlicerDynamicForPlugin () : Plugin("Slicer for Loops using a dynamic policy",1) {}
	~SlicerDynamicForPlugin () {}

	void init ()
        {
	   sys.registerSlicer("DynamicFor", new SlicerDynamicFor() );	
	}
};


}
}

nanos::ext::SlicerDynamicForPlugin NanosXPlugin;
