#ifndef _NANOS_PLUGIN
#define _NANOS_PLUGIN

#include <dlfcn.h>
#include <string>
#include <iostream>

namespace nanos {

class Plugin {
  private:
	std::string name;
	int    version;
	void  *handler;
  public:
	virtual void init() {};
	virtual void fini() {};
};

class PluginManager {
  private:
	static std::string pluginsDir;
  public:

	static void setDirectory (const char *dir) { pluginsDir = dir; }
	static void setDirectory (const std::string & dir) { pluginsDir = dir; }
	static bool load(const char *plugin_name);
};

}

#endif
