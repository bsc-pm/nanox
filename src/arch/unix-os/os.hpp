#ifndef _NANOS_OS
#define _NANOS_OS

#include <string>
#include <vector>
#include <stdlib.h>

namespace nanos {

// this is UNIX-like OS
// TODO: ABS and virtualize
class OS
{
public:
	class Argument {
	friend class OS;
	public:
		char * name;
		int nparam;
	public:
		Argument(char *arg,int i) : name(arg),nparam(i) {}
		char * getName() const { return name; }
	};

	//TODO: make it autovector?
	typedef std::vector<Argument *> ArgumentList;

	static char **argv;
	static long *argc;

	static const char *getEnvironmentVariable(const std::string &variable);
	static void getProgramArguments (ArgumentList &list);
	static void consumeArgument (Argument &arg);
	static void repackArguments ();
};

// inlined functions

inline const char * OS::getEnvironmentVariable (const std::string &name)
{
	return getenv(name.c_str());
}

};


#endif

