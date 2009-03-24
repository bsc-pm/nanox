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
	public:
		Argument(char *arg) : name(arg) {}
		char * getName() const { return name; }
	};

	//TODO: make it autovector?
	typedef std::vector<Argument *> ArgumentList;

	static const char *getEnvironmentVariable(const std::string &variable);
	static void getProgramArguments (ArgumentList &list);
	static void consumeArgument (Argument &arg);
};

// inlined functions

inline const char * OS::getEnvironmentVariable (const std::string &name)
{
	return getenv(name.c_str());
}

};


#endif
