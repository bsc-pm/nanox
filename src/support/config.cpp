#include <iostream>
#include <sstream>
#include <string>
#include "config.hpp"
#include "os.hpp"
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <ext/functional>
#include <functional>
#include "debug.hpp"

using namespace nanos;

void Config::setDefaults()
{
}

void Config::parseFiles ()
{
}

void Config::registerEnvOption (Option *opt)
{
	envOptions.push_back(opt);
}

void Config::registerArgOption (Option *opt)
{
	argOptions[opt->getName()] = opt;
}

void Config::parseEnvironment ()
{
	for ( OptionList::iterator it = envOptions.begin();
		it < envOptions.end(); it++ )
	{
		Option &opt = **it;

		const char *env = OS::getEnvironmentVariable(opt.getName());
		if (!env) continue;

		const std::string tmp(env);
		std::istringstream iss(env);

		try {
			opt.parse(env);
		} catch (InvalidOptionException &exception) {
			std::cerr << "WARNING:" << exception.what() << std::endl;
		}
	}
}

// C-strings are used in this function because std::string substring 
// and character removal are O(N) compared to the possible O(1) 
// with C style pointer manipulation.
// Even so, a std::string needs to be constructed to access the argument map
// so it's not clear if its worth it.
void Config::parseArguments ()
{
	OS::ArgumentList list;
	OS::getProgramArguments(list);

	for (OS::ArgumentList::iterator it = list.begin();
		it < list.end(); it++)
	{

		char * arg((*it)->name);
		char * value=0;
		bool needValue=true;

		if ( arg[0] != '-' ) continue;

		arg++;
		// support --args
		if ( arg[0] == '-' ) arg++;

		if ( (value = strchr(arg,'=')) != NULL ) {
			// -arg=value form
			*value = 0; // sepparate arg from value
			value++; // point to the beginning of value
			needValue = false;
		} else {
			// -arg value form
		}
		
		OptionMap::iterator obj = argOptions.find(std::string(arg));

		if ( obj != argOptions.end()) {
			Option &opt = *(*obj).second;

			if ( needValue && opt.getType() != Option::FLAG ) {
				OS::consumeArgument(*(*it));
				it++;
				if ( it == list.end() ) 
					throw InvalidOptionException(opt,"");
				value = (*it)->name;
			} 
			try {
				opt.parse(value);
			} catch (InvalidOptionException &exception) {
				std::cerr << "WARNING:" << exception.what() << std::endl;
			}
			OS::consumeArgument(*(*it));
		}
	}

	OS::repackArguments();
}

void Config::init ()
{
	setDefaults();
	parseFiles();
	parseEnvironment();
	parseArguments();
}

//TODO: move to utility header
template<typename T>
void deleter(T *p) { delete p; }

template<typename T>
void pair_deleter1 (std::pair<std::string,T *> pair) { delete pair.first; }

template<typename T>
void pair_deleter2 (std::pair<std::string,T *> pair) { delete pair.second; }

template<typename T>
T * creator (T *p) { return new T(*p); }

template<typename T>
T * cloner (T *p) { return p->clone(); }

void Config::clear ()
{
      std::for_each(envOptions.begin(),envOptions.end(),deleter<Option>);
      std::for_each(argOptions.begin(),argOptions.end(),pair_deleter2<Option>);
      envOptions.clear();
      argOptions.clear();
}

//TODO: generalize?
class map_copy {
Config::OptionMap& dest;
public:

map_copy(Config::OptionMap &d) : dest(d) {}
void operator()(Config::OptionMap::value_type pair) {  dest[pair.first] = pair.second->clone(); }
};

void Config::copy (const Config &cfg)
{
     std::transform(cfg.envOptions.begin(), cfg.envOptions.end(), envOptions.begin(),
		     cloner<Option> );
     std::for_each(cfg.argOptions.begin(), cfg.argOptions.end(), map_copy(argOptions));
}

Config::Config (const Config &cfg)
{
  copy(cfg);
}

const Config & Config::operator= (const Config &cfg)
{
     // handle self-assignment
     if ( this == &cfg ) return *this;
     
     clear();
     copy(cfg);
     return *this;
}

Config::~Config ()
{
    clear();
}

/** Options **/

