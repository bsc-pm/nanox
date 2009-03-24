#include "workdescriptor.hpp"
#include "processingelement.hpp"
#include <stdarg.h>
#include <stdio.h>
#include <stdexcept>
#include <string.h>

using namespace nanos;

bool SimpleWD::canRunIn(PE &pe)
{
	return pe.getArchitecture() == architecture;
}

// void WD::setArgument ()
// {
// }

//TODO: interface is too cumbersome (redo)
void WorkData::setArguments( int total_size, int nrefs, int nvals, ... )
{
  if ( total_size > (signed)(64*sizeof(void *)) ) {
	throw std::runtime_error("WD argument overflow not implemented yet");
  }

  int idx=0,pidx=0;;
  va_list vargs;
  va_start(vargs, nvals);


  while ( nrefs-- > 0 ) {
        void * arg = va_arg(vargs, void *);
	data[idx] = arg;
	positions[pidx].first = idx++;
	positions[pidx].second = sizeof(void *);
	pidx++;
  }

  while ( nvals-- > 0 ) {
      size_t size = va_arg(vargs, size_t);
      void * arg = va_arg(vargs, void *);

      memcpy(&data[idx],arg,size);
      positions[pidx].first = idx;
      positions[pidx].second = size;
      pidx++;
      // add size rounded up
      idx += (size+sizeof(void *)-1)/sizeof(void *);
  }

  va_end(vargs);
}