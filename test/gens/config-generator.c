#include <stdio.h>

int main()
{
   printf("exec_versions=\"normal cilk\"\n");
   printf("test_ENV_normal=\"NX_ARGS=''\"\n");
   printf("test_ENV_cilk=\"NX_ARGS='--schedule cilk'\"\n");
}

