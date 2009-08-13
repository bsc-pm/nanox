#include "nanos.h"
#include <stdlib.h>
#include <stdio.h>

void nanos_handle_error (nanos_err_t err)
{
   switch ( err ) {
     default:
     case NANOS_UNKNOWN_ERR:
          fprintf(stderr,"Unkown NANOS error decteded\n");
	  break;
     case NANOS_UNIMPLEMENTED: 
          fprintf(stderr,"Requested NANOS service not implemented\n");
	  break;  
   }
   abort();
}
