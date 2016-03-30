**Back to:** [README](../../README.md) > [developers guide](../developers_guide.md) >

Nanos++ Application Progamming Interface
========================================

Altough Nanos++ Runtime Library is not intended for programmers to write
applications calling its services directly, we believe it is a valuable
information for programmers, compiler developers and for debugging purposes.
All services and types are documented in this manual:

* Nanos++ API: description, types and services
* Nanos++ C API. By default all Nanos++ common services has been programmed
  in C, so this is the common interface.
* Nanos++ Fortran API. All previous C services have an alias which allow alsoto
  use them in Fortran application.

````cpp
  __attribute__((alias nanos_service))) void nanos_service_ ( nanos_type_t nanos );

  void nanos_service ( nanos_type_t nanos )
  {
     // Service definition
  }
````

Interface Usage
---------------

* All types and services are defined in the **nanos.h** header.
* Use `-INANOX_INSTALL_PREFIX/include` in the compilation step.
* Use `-LNANOX_INSTALL_PREFIX/lib/\<version\> -lnanox-c` in the linkage step.
  Where version can be:
   * **performance**: the default  implementation.
   * **debug**: this version also includes debug information.
   * **instrumentation**: raises runtime internal events which can be managed
     through one the instrumentation plugins.
   * **instrumentation-debug**: a combination of debug and instrumentation
     versions.
* The API offers a default error handler `nanos_handle_error(nanos_err_t err)`.
  List of return values:
  * `NANOS_OK`: Everything was ok.
  * `NANOS_UNKNOW_ERR`: Not specified error.
  * `NANOS_UNIMPLEMENTED`: This service is not implemented yet.
* Easier to use surrounding all library calls with the `NANOS_SAFE` macro. E.g.:

````cpp
NANOS_SAFE(nanos_wd_create(...));
````

* `NANOS_SAFE` will try to execute and it will handle the error (if any):

````cpp
#define NANOS_SAFE( call ) \
do {\
   nanos_err_t err = call;\
   if ( err != NANOS_OK ) nanos_handle_error( err );\
} while (0)
````
