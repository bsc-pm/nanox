#####################################################################################
#      Copyright 2009 Barcelona Supercomputing Center                               #
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
#      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             #
#####################################################################################

SUBDIRS = performance debug instrumentation instrumentation-debug

include $(top_srcdir)/src/common.am

#lib_LTLIBRARIES=libnanox-c.la

pkginclude_HEADERS=\
	nanos.h \
   nanos_version.h \
   nanos_reduction.h \
   nanos_atomic.h \
	$(END)

BUILT_SOURCES=\
   nanos_version.h

EXTRA_DIST=\
   nanos_version.def

CLEANFILES=nanos_version.h

nanos_version.h : $(srcdir)/nanos_version.def
	$(AM_V_GEN) (\
       echo "#ifndef  _NANOS_VERSION_H_" > $@; \
       echo "#define  _NANOS_VERSION_H_" >> $@; \
       cat $(srcdir)/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#define NANOS_API_\U\1 \2,' >> $@; \
       echo "#ifdef _MERCURIUM" >> $@; \
       cat $(srcdir)/nanos_version.def | $(SED) -e 's,^\s*\(\S\+\)\s*=\s*\(\S\+\)\s*$$,#pragma nanos interface family(\1) version(\2),' >> $@; \
       echo "#endif // _MERCURIUM" >> $@; \
       echo "#endif // _NANOS_VERSION_H_" >> $@; \
       )
