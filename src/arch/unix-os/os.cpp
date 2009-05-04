#include "os.hpp"
#include <stdlib.h>

using namespace nanos;

void OS::getProgramArguments (ArgumentList &list)
{
	long *p;
	int i;

	int argc;
	char **argv;

	// variables are before environment
	p=(long *)environ;

	// go backwards until we find argc
	p--;
	for ( i = 0 ; *(--p) != i; i++ );

	argc = i;
	argv = (char **) p+1;

	// build vector
	for ( i = 0; i < argc; i++)
		list.push_back(new Argument(argv[i]));

}

void OS::consumeArgument (Argument &arg)
{
	*arg.name = 0;
}

