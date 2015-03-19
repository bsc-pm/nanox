# ===========================================================================
#   http://www.gnu.org/software/autoconf-archive/ax_check_compile_flag.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_OPENCL(FLAG, [ACTION-SUCCESS], [ACTION-FAILURE], [EXTRA-FLAGS], [INPUT])
#
# DESCRIPTION
#
#   Check whether OpenCL path to the headers and libraries are correctly specified.
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
#
# LICENSE
#
#   Copyright (c) 2015 Jorge Bellon <jbellon@bsc.es>
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_CHECK_OPENCL],
[
# AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX
# AS_VAR_PUSHDEF([CACHEVAR],[ax_cv_check_[]_AC_LANG_ABBREV[]flags_$4_$1])dnl
# AC_CACHE_CHECK([whether _AC_LANG compiler accepts $1], CACHEVAR, [
#   ax_check_save_flags=$[]_AC_LANG_PREFIX[]FLAGS
#   _AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $4 $1"
#   AC_COMPILE_IFELSE([m4_default([$5],[AC_LANG_PROGRAM()])],
#     [AS_VAR_SET(CACHEVAR,[yes])],
#     [AS_VAR_SET(CACHEVAR,[no])])
#   _AC_LANG_PREFIX[]FLAGS=$ax_check_save_flags])
# AS_IF([test x"AS_VAR_GET(CACHEVAR)" = xyes],
#   [m4_default([$2], :)],
#   [m4_default([$3], :)])
# AS_VAR_POPDEF([CACHEVAR])dnl
AC_PREREQ(2.59)dnl for _AC_LANG_PREFIX

#Check if an OpenCL implementation is installed.
AC_ARG_WITH(opencl,
[AS_HELP_STRING([--with-opencl=PATH],
                [specify prefix directory for installed OpenCL package.
                 Equivalent to --with-opencl-include=PATH/include
                 plus --with-opencl-lib=PATH/lib])])
AC_ARG_WITH(opencl-include,
[AS_HELP_STRING([--with-opencl-include=PATH],
                [specify directory for installed OpenCL include files])])
AC_ARG_WITH(opencl-lib,
[AS_HELP_STRING([--with-opencl-lib=PATH],
                [specify directory for the installed OpenCL library])])

# If the user specifies --with-opencl, $with_opencl value will be 'yes'
#                       --without-opencl, $with_opencl value will be 'no'
#                       --with-opencl=somevalue, $with_opencl value will be 'somevalue'
if test "x$with_opencl" != xyes -a "x$with_opencl" != xno; then
  openclinc="$with_opencl/include"
  AC_CHECK_FILE([$with_opencl/lib64],
    [opencllib=$with_opencl/lib64],
    [opencllib=$with_opencl/lib])
fi
if test "x$with_opencl_include" != x; then
  openclinc="$with_opencl_include"
fi
if test "x$with_opencl_lib" != x; then
  opencllib="$with_opencl_lib"
fi

# This is fulfilled even if $with_opencl="yes" 
# This happens when user leaves --with-value empty
# In this case, both openclinc and opencllib will be empty
# so the test should search in default locations and LD_LIBRARY_PATH
if test "x$with_opencl" != xno -a "x$with_opencl$with_opencl_include$with_opencl_lib" != x; then
    #tests if provided headers and libraries are usable and correct
    bak_CFLAGS="$CFLAGS"
    bak_CxXFLAGS="$CXXFLAGS"
    bak_CPPFLAGS="$CPPFLAGS"
    bak_LIBS="$LIBS"
    bak_LDFLAGS="$LDFLAGS"

    CFLAGS=
    CXXFLAGS=
    CPPFLAGS=-I$openclinc
    LIBS=
    LDFLAGS=-L$opencllib

    # One of the following two header files has to exist
    AC_CHECK_HEADERS([CL/opencl.h OpenCL/opencl.h], [opencl=yes; break])
    # Look for clGetPlatformIDs function in either libmali.so or libOpenCL.so libraries
    if test x$opencl = xyes; then
        AC_SEARCH_LIBS([clGetPlatformIDs],
                  [mali OpenCL],
                  [opencl=yes],
                  [opencl=no])
    fi

    if test x$opencl = xyes; then
        AC_MSG_CHECKING([OpenCL version])
        AC_RUN_IFELSE(
               [AC_LANG_PROGRAM(
                 [
                    #ifdef HAVE_CL_OPENCL_H
                        #include <CL/opencl.h>
                    #else if HAVE_OPENCL_OPENCL_H
                        #include <OpenCL/opencl.h>
                    #endif
                    #include <stdio.h>
                    #include <stdlib.h>

                    cl_int err;
                    cl_platform_id platform = 0;
                    cl_device_id device = 0;
                    size_t len;
                    char *ocl_ver;
                    int ret = 0;],
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
                    return ret;
                 ])],
               [oclversion=$(cat conftest.out)
                AC_MSG_RESULT([$oclversion])
                oclversion=$(expr "x$oclversion" : 'xOpenCL [a-zA-Z\+]* \(.*\)$')
               ],
               [AC_MSG_FAILURE([OpenCL version test execution failed])])
    fi


    CFLAGS="$bak_CFLAGS"
    CPPFLAGS="$bak_CPPFLAGS"
    LIBS="$bak_LIBS"
    LDFLAGS="$bak_LDFLAGS"

    if test x$opencl != xyes; then
        AC_MSG_ERROR([OpenCL was not found. Please, check that the provided directories are correct.])
    fi
fi

if test x$opencl = xyes; then
    ARCHITECTURES="$ARCHITECTURES opencl"

    AC_DEFINE([OpenCL_DEV],[],[Indicates the presence of the OpenCL arch plugin.])
    AC_SUBST([openclinc])
    AC_SUBST([opencllib])

    AC_DEFINE([CL_USE_DEPRECATED_OPENCL_2_0_APIS],[],[Disables warnings when using functions deprecated in OpenCL 2.0])
fi

AM_CONDITIONAL([OPENCL_SUPPORT],[test x$opencl = xyes])

AC_SUBST([opencl])
])dnl AX_CHECK_OPENCL
