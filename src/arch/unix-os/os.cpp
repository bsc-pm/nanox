#include "os.hpp"
#include <stdlib.h>

using namespace nanos;

long *  OS::argc  = 0;
char ** OS::argv = 0;

void OS::getProgramArguments (ArgumentList &list)
{
	long *p;
	int i;

    if ( !argc ) {
	  // variables are before environment
      p=(long *)environ;

	  // go backwards until we find argc
	  p--;
	  for ( i = 0 ; *(--p) != i; i++ );

	  argc = p;
	  argv = (char **) p+1;
    }

	// build vector
  	list.reserve(*argc);
 	for ( i = 0; i < *argc; i++)
 		list.push_back(new Argument(argv[i],i));

}

void OS::consumeArgument (Argument &arg)
{
	argv[arg.nparam] = 0;
}

void OS::repackArguments ()
{
	int i,hole = 0;

        // find first hole
	for ( i  = 0; i < *argc; i++ )
	    if (!argv[i]) {
		hole=i++;
		break;
            }

	for ( ; i < *argc; i++ )
	    if (argv[i]) {
		argv[hole]=argv[i];
		argv[i]=0;
		hole++;
	    }

	if (hole != 0)
	   *argc = hole;
}

void * OS::loadDL(std::string &dir, std::string &name)
{
   std::string filename;
   filename = dir + "/" + name + ".so";
   /* open the module */
   return dlopen ( filename.c_str(), RTLD_GLOBAL | RTLD_NOW );
}

void * OS::dlFindSymbol(void *dlHandler, std::string &symbolName)
{
   return dlsym ( dlHandler, symbolName.c_str() );
}

void * OS::dlFindSymbol(void *dlHandler, const char *symbolName)
{
   return dlsym ( dlHandler, symbolName );
}

