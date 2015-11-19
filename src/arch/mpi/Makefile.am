#####################################################################################
#      Copyright 2015 Barcelona Supercomputing Center                               #
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

include $(top_srcdir)/src/common.am

pkginclude_HEADERS=\
	nanos-mpi.h \
	$(END)

devinclude_HEADERS = \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidd.hpp \
	mpithread.hpp \
	$(END)

pe_mpi_sources = \
	mpiplugin.cpp \
	$(END)

mpiinclude_HEADERS = \
	nanos-mpi.h \
	mpi-api.cpp \
	mpiall.cpp \
	mpiplugin.cpp \
	mpiprocessor_fwd.hpp \
	mpiprocessor_decl.hpp \
	mpiprocessor.hpp \
	mpiprocessor.cpp \
	mpiremotenode_fwd.hpp \
	mpiremotenode_decl.hpp \
	mpiremotenode.hpp \
	mpiremotenode.cpp \
	mpidevice_decl.hpp \
	mpidevice.hpp \
	mpidevice.cpp \
	mpidd.hpp \
	mpidd.cpp \
	mpithread.hpp \
	mpithread.cpp \
	$(END)

mpiincludedir = $(prefix)/src/arch/mpi

noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =


test_cxxflags = $(common_includes) $(bin_cxxflags)

install-data-hook:
	echo -n "${test_cxxflags}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > $(DESTDIR)$(mpiincludedir)/test_flags
	echo " -DMPICH_IGNORE_CXX_SEEK -I${devincludedir}" >> $(DESTDIR)$(mpiincludedir)/test_flags
	echo $(CXX) > $(DESTDIR)$(mpiincludedir)/nanox_compiler
if is_debug_enabled
	echo -n "${common_debug_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > $(DESTDIR)$(mpiincludedir)/debug_flags
	echo " -DMPICH_IGNORE_CXX_SEEK -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/debug_flags
endif
if is_instrumentation_enabled
	echo -n "${common_instrumentation_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > $(DESTDIR)$(mpiincludedir)/instrumentation_flags
	echo " -DMPICH_IGNORE_CXX_SEEK -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/instrumentation_flags
endif
if is_instrumentation_debug_enabled
	echo -n "${common_instrumentation_debug_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags
	echo " -DMPICH_IGNORE_CXX_SEEK -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags
endif
if is_performance_enabled
	echo -n "${common_performance_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > $(DESTDIR)$(mpiincludedir)/performance_flags
	echo " -DMPICH_IGNORE_CXX_SEEK -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/performance_flags
endif


uninstall-hook:
	rm -f $(DESTDIR)$(mpiincludedir)/test_flags
	rm -f $(DESTDIR)$(mpiincludedir)/nanox_compiler
	rm -f $(DESTDIR)$(mpiincludedir)/debug_flags
	rm -f $(DESTDIR)$(mpiincludedir)/instrumentation_flags
	rm -f $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags
	rm -f $(DESTDIR)$(mpiincludedir)/performance_flags
