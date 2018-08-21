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

SUBDIRS = . src tests doc scripts
DIST_SUBDIRS = tests doc src scripts
EXTRA_DIST = ChangeLog

ACLOCAL_AMFLAGS=-I m4

devincludedir = $(prefix)/include/nanox-dev
nodist_devinclude_HEADERS = nanox-config.h

# We use an install hook to remove old deprecated libraries, which are not tracked
# anymore by Automake so they need to be removed manually
REMOVE_LIBS = \
	      libnanox-ss \
	      libnanox-sched-affinity-smartpriority \
	      $(END)

remove-old-libraries:
	@for oldlib in $(REMOVE_LIBS); do \
	    for version in performance debug instrumentation instrumentation-debug; do \
	        for f in $(libdir)/$${version}/$${oldlib}*; do \
	            if test -e $${f}; then \
	                echo "Removing old library $${f}"; \
	                rm -f $${f}; \
	            fi \
	        done \
	    done \
	done

REMOVE_CFGS = \
	      paraver_configs/hwc/papi/performance/ipc.cfg \
	      $(END)

remove-old-cfgs:
	@for cfg in $(REMOVE_CFGS); do \
	    for f in $(docdir)/$${cfg}; do \
	        if test -e $${f}; then \
	            echo "Removing old cfg $${f}"; \
	            rm -f $${f}; \
	        fi \
	    done \
	done


install-exec-hook: remove-old-libraries remove-old-cfgs

rpm: dist-gzip
	$(MAKE) -C scripts rpm

deb: dist-gzip
	$(MAKE) -C scripts deb

dist-hook:
	if [ -x "$(GIT)" ]; \
	then \
	    "$(GIT)" --git-dir=$(top_srcdir)/.git log --pretty=format:"[%h] %cd : %s (%an)" --date=short > $(distdir)/ChangeLog; \
	    echo "@NANOX_BUILD_VERSION@" > ${distdir}/VERSION; \
	elif [ -e $(top_srcdir)/VERSION ]; \
	then \
	    cp $(top_srcdir)/VERSION $(distdir)/VERSION; \
	fi
