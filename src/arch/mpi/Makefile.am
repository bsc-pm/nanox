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
	mpiicpc_syms.cpp \
	hostinfo.hpp \
	hostinfo.cpp \
	$(END)


noinst_LTLIBRARIES =
debug_LTLIBRARIES =
instrumentation_LTLIBRARIES =
instrumentation_debug_LTLIBRARIES =
performance_LTLIBRARIES =

mpiincludedir = $(prefix)/src/arch/mpi

test_cxxflags = $(common_includes) $(bin_cxxflags)

version_flags=
if is_debug_enabled
version_flags+= debug_flags
endif
if is_instrumentation_enabled
version_flags+= instrumentation_flags
endif
if is_instrumentation_debug_enabled
version_flags+= instrumentation_debug_flags
endif
if is_performance_enabled
version_flags+= performance_flags
endif

all: nanos_compiler test_flags $(version_flags)

	$(DESTDIR)$(mpiincludedir)/test_flags

nanox_compiler:
	echo $(CXX) > nanox_compiler

tests_flags:
	echo -n "${test_cxxflags} -I${devincludedir}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g" > test_flags

debug_flags:
	echo -n "${common_debug_CPPFLAGS} -I$(DESTDIR)$(mpiincludedir) ${common_debug_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g; s/config.h/nanos-config.h/" > $(DESTDIR)$(mpiincludedir)/debug_flags
	echo " -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/debug_flags

instrumentation_flags:
	echo -n "${common_instrumentation_CPPFLAGS} -I$(DESTDIR)$(mpiincludedir) ${common_instrumentation_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g; s/config.h/nanos-config.h/" > $(DESTDIR)$(mpiincludedir)/instrumentation_flags
	echo " -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/instrumentation_flags

instrumentation_debug_flags:
	echo -n "${common_instrumentation_debug_CPPFLAGS} -I$(DESTDIR)$(mpiincludedir) ${common_instrumentation_debug_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g; s/config.h/nanos-config.h/" > $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags
	echo " -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags

performance_flags:
	echo -n "${common_performance_CPPFLAGS} -I$(DESTDIR)$(mpiincludedir) ${common_performance_CXXFLAGS}" | sed "s@${top_srcdir}@${abs_top_srcdir}@g; s/config.h/nanos-config.h/" > $(DESTDIR)$(mpiincludedir)/performance_flags
	echo " -I${devincludedir} -DNANOX_PREFIX=\"$(prefix)\"" >> $(DESTDIR)$(mpiincludedir)/performance_flags

install-data-hook: install-nanox_compiler $(addprefix,install-,$(version_flags)) install-nanos_config

install-nanox_compiler: nanox_compiler
	$(install_sh_DATA) nanox_compiler $(DESTDIR)$(mpiincludedir)

install-nanos_config: $(abs_top_builddir)/config.h
	$(install_sh_DATA) $(abs_top_builddir)/config.h $(DESTDIR)$(mpiincludedir)/nanos-config.h

install-test_flags:
	$(install_sh_DATA) test_flags $(DESTDIR)$(mpiincludedir)

install-debug_flags:
	$(install_sh_DATA) debug_flags $(DESTDIR)$(mpiincludedir)

install-instrumentation_flags:
	$(install_sh_DATA) instrumentation_flags $(DESTDIR)$(mpiincludedir)

install-instrumentation_debug_flags:
	$(install_sh_DATA) instrumentation_debug_flags $(DESTDIR)$(mpiincludedir)

install-performance_flags:
	$(install_sh_DATA) performance_flags $(DESTDIR)$(mpiincludedir)

uninstall-hook:
	rm -f $(DESTDIR)$(mpiincludedir)/test_flags
	rm -f $(DESTDIR)$(mpiincludedir)/nanox_compiler
	rm -f $(DESTDIR)$(mpiincludedir)/debug_flags
	rm -f $(DESTDIR)$(mpiincludedir)/instrumentation_flags
	rm -f $(DESTDIR)$(mpiincludedir)/instrumentation_debug_flags
	rm -f $(DESTDIR)$(mpiincludedir)/performance_flags

