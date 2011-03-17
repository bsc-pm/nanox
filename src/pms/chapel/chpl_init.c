#include "nanos-int.h"

void nanos_chapel_pre_init ( void * );

__attribute__((weak, section( "nanos_init" ))) nanos_init_desc_t __chpl_init = { nanos_chapel_pre_init, NULL };

