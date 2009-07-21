#include "debug.hpp"
#include "plugin.hpp"

using namespace nanos;

std::string PluginManager::plugin_dir(".");

bool PluginManager::load (const char *name)
{
    std::string filename;
    void * handler; 

    //TODO: check if already loaded

    verbose0("trying to load plugin " << name);

    filename = plugin_dir+"/nanox-"+name+".so";
    /* open the module */
    handler = dlopen(filename.c_str(),RTLD_GLOBAL | RTLD_NOW);
    if (!handler) {
	verbose0("plugin error=" << dlerror());
	return false;
    }

    Plugin *plugin = (Plugin *) dlsym(handler,"NanosXPlugin");
    if (!plugin) {
        verbose0("plugin error=" << dlerror());
	return false;
    }
    
    plugin->init();

    //TODO: register

    return true;
}
