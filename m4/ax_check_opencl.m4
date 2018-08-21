#####################################################################################
#      Copyright 2009-2018 Barcelona Supercomputing Center                          #
#                                                                                   #
#      This file is part of the NANOS++ library.                                    #
#                                                                                   #
#      NANOS++ is free software: you can redistribute it and/or modify              #
#      it under the terms of the GNU Lesser General Public License as published by  #
#      the Free Software Foundation, either version 3 of the License, or            #
#      (at your option) any later version.                                          #
#                                                                                   #
#      NANOS++ is distributed in the hope that it will be useful,                   #
#      but WITHOUT ANY WARRANTY; without even the implied warranty of               #
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
#      GNU Lesser General Public License for more details.                          #
#                                                                                   #
#      You should have received a copy of the GNU Lesser General Public License     #
#      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            #
#####################################################################################

#
# SYNOPSIS
#
#   AX_CHECK_OPENCL
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#   Also checks that the library version is OpenCL 1.1 or greater.
#

AC_DEFUN([AX_CHECK_OPENCL],
[
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

#Check if an OpenCL implementation is installed.
AC_ARG_WITH(opencl,
  [AS_HELP_STRING([--with-opencl,--with-opencl=PATH],
                [search in system directories or specify prefix directory for installed OpenCL package.])],
  [
    # Check if the user provided a valid PATH
    AS_IF([test -d "$withval"],[
      opencl=yes
      opencl_path_provided=yes
    ],[
      opencl=$withval
      opencl_path_provided=no
    ])dnl
  ],[
    # Default: check if opencl is available
    opencl=yes
    opencl_path_provided=no
  ])

AC_ARG_WITH(opencl-include,
[AS_HELP_STRING([--with-opencl-include=PATH],
                [specify directory for installed OpenCL include files])])
AC_ARG_WITH(opencl-lib,
[AS_HELP_STRING([--with-opencl-lib=PATH],
                [specify directory for the installed OpenCL library])])

# Check if user specifically requested this package.
# It will generate errors if we are not able to find headers/libs
AS_IF([test "x$with_opencl$with_opencl_include$with_opencl_lib" != x],[
  user_requested="yes"
],[
  user_requested="no"
])dnl

# If the user specifies --with-opencl, $with_opencl value will be 'yes'
#                       --without-opencl, $with_opencl value will be 'no'
#                       --with-opencl=somevalue, $with_opencl value will be 'somevalue'
AS_IF([test "$opencl_path_provided" = yes],[
  openclinc="-isystem $with_opencl/include -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
  AS_IF([test -d $with_opencl/lib64],
    [opencllib="-L$with_opencl/lib64 -Wl,-rpath,$with_opencl/lib64"],
    [opencllib="-L$with_opencl/lib -Wl,-rpath,$with_opencl/lib"])dnl
])dnl

AS_IF([ test $with_opencl_include],[
  openclinc="-isystem $with_opencl_include -DCL_USE_DEPRECATED_OPENCL_1_2_APIS"
])dnl

AS_IF([test $with_opencl_lib],[
  opencllib="-L$with_opencl_lib -Wl,-rpath,$with_opencl_lib"
])dnl

# This is fulfilled even if $with_opencl="yes" 
# This happens when user leaves --with-value empty
# In this case, both openclinc and opencllib will be empty
# so the test should search in default locations and LD_LIBRARY_PATH
AS_IF([test "x$with_opencl" != xno],[
    AC_LANG_PUSH([C++])

    #tests if provided headers and libraries are usable and correct
    AX_VAR_PUSHVALUE([CPPFLAGS],[$CPPFLAGS $openclinc])
    AX_VAR_PUSHVALUE([CXXFLAGS],[$CXXFLAGS -Wno-error=comment])
    AX_VAR_PUSHVALUE([LDFLAGS],[$LDFLAGS $opencllib])
    AX_VAR_PUSHVALUE([LIBS])

    # One of the following two header files has to exist
    AC_CHECK_HEADERS([CL/opencl.h OpenCL/opencl.h], [opencl=yes; break], [opencl=no])
    # Look for clGetPlatformIDs function in either libmali.so or libOpenCL.so libraries
    AS_IF([test x$opencl = xyes],[
        AC_SEARCH_LIBS([clGetPlatformIDs],
                  [mali OpenCL],
                  [opencl=yes],
                  [opencl=no])
    ])

    AS_IF([test "x$opencl" = xyes],[
      AC_CACHE_CHECK([OpenCL version],[ac_cv_opencl_version],
        [AC_RUN_IFELSE(
          [AC_LANG_PROGRAM(
            [
               #ifdef HAVE_CL_OPENCL_H
                   #include <CL/opencl.h>
               #elif HAVE_OPENCL_OPENCL_H
                   #include <OpenCL/opencl.h>
               #endif
               #include <stdio.h>
               #include <stdlib.h>

               cl_int err;
               cl_platform_id platform = 0;
               cl_device_id device = 0;
               size_t len;
               char *ocl_ver;
               int ret = 0;
            ],
            [
               /* Setup OpenCL environment. */
               err = clGetPlatformIDs(1, &platform, NULL);
               if (err != CL_SUCCESS) {
                   printf( "clGetPlatformIDs() failed with %d\n", err );
                   return 1;
               }
               
               err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
               if (err != CL_SUCCESS) {
                   printf( "clGetDeviceIDs() failed with %d\n", err );
                   return 1;
               }
               
               err = clGetDeviceInfo(device,  CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &len);
               ocl_ver = (char *)malloc(sizeof(char)*len);
               err = clGetDeviceInfo(device,  CL_DEVICE_OPENCL_C_VERSION, len, ocl_ver, NULL);


               FILE* out = fopen("conftest.out","w");
               fprintf(out,"%s\n", ocl_ver);
               fclose(out);
               free(ocl_ver);

               err = clGetDeviceInfo(device,  CL_DEVICE_VERSION, 0, NULL, &len);
               ocl_ver = (char *)malloc(sizeof(char)*len);
               err = clGetDeviceInfo(device,  CL_DEVICE_VERSION, len, ocl_ver, NULL);
			
               FILE* out_device_version = fopen("conftest_device_version.out","w");
               fprintf(out_device_version,"%s\n", ocl_ver);
               fclose(out_device_version);
               
               free(ocl_ver);
               return ret;
            ])],
          [ac_cv_opencl_version=$(cat conftest.out),
	   ac_cv_opencl_device=$(cat conftest_device_version.out)
          ],
          [
            AS_IF([ $(cat conftest_device_version.out | grep -q Altera) ],[
               ac_cv_opencl_version=$(cat conftest.out),
               ac_cv_opencl_device=$(cat conftest_device_version.out)
            ],
            [
               AS_IF([test "x$user_requested" = xyes],
               [
                  AC_MSG_FAILURE([
------------------------------------
OpenCL version test execution failed
------------------------------------])
               ],[
                  opencl=no
               ])
             ])
          ],
          [
            ac_cv_opencl_version=skip
          ])
        ])
	
      #Check if is Altera OpenCL
      AS_IF([test "$ac_cv_opencl_version" != skip],[
	AS_IF([$(echo $ac_cv_opencl_device | grep -q Altera)],[ac_cv_opencl_altera=yes],[ac_cv_opencl_altera=no])
      ])dnl

      AS_IF([test "$ac_cv_opencl_version" != skip],[
        ac_cv_opencl_version=$(expr "$ac_cv_opencl_version" : "OpenCL C \(@<:@\.0-9@:>@\+\)")
      ])dnl




    ])dnl opencl


    opencllibs="$LIBS"

    AX_VAR_POPVALUE([CPPFLAGS])
    AX_VAR_POPVALUE([CXXFLAGS])
    AX_VAR_POPVALUE([LDFLAGS])
    AX_VAR_POPVALUE([LIBS])

    AC_LANG_POP([C++])

    AS_IF([test "$user_requested" = yes -a "$opencl" != yes],[
        AC_MSG_ERROR([
------------------------------
OpenCL path was not correctly specified. 
Please, check that the provided directories are correct.
------------------------------])
    ])dnl

    AS_IF([test "$ac_cv_opencl_version" = skip],[
        AC_MSG_WARN([
------------------------------
OpenCL library version cannot be checked
because cross-compilation mode has been detected.
------------------------------])
    ],[
      AX_COMPARE_VERSION([$ac_cv_opencl_version],[lt],[1.1],[opencl=no])
      
      AS_IF([test "$user_requested" = yes -a "$opencl" != yes -a "$ac_cv_opencl_altera" != yes  ],[
        AC_MSG_ERROR([
------------------------------
Version of the provided OpenCL package is too old.
OpenCL 1.1 or greater is required.
------------------------------])
      ])dnl
    ])dnl

])dnl with_opencl

AS_IF([test "$ac_cv_opencl_altera" = yes], [opencl=yes])

AS_IF([test $opencl = yes],[
    ARCHITECTURES="$ARCHITECTURES opencl"

    AC_DEFINE([OpenCL_DEV],[],[Indicates the presence of the OpenCL arch plugin.])
    AC_DEFINE([CL_USE_DEPRECATED_OPENCL_2_0_APIS],[],[Disables warnings when using functions deprecated in OpenCL 2.0])
    AC_SUBST([HAVE_OPENCL], [OpenCL_DEV])
], [
    AC_SUBST([HAVE_OPENCL], [NO_OpenCL_DEV])
])dnl


AM_CONDITIONAL([OPENCL_SUPPORT],[test "$opencl" = yes])

AC_SUBST([opencl])
AC_SUBST([openclinc])
AC_SUBST([opencllib])
AC_SUBST([opencllibs])

])dnl AX_CHECK_OPENCL
