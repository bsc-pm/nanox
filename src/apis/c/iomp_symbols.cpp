#include <stdint.h>

extern "C" {

// Minimal support of -openmp
void __kmpc_begin(void* loc, int32_t flags);
void __kmpc_end(void* loc);

void __kmpc_begin(void* loc, int32_t flags) { /* nop */ }
void __kmpc_end(void* loc) { /* nop */ }

}
