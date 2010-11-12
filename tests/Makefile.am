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

SUBDIRS = gens

CLEANFILES=

CLEANFILES+=tests.log

performance:
	$(MAKE) -C gens performance

debug:
	$(MAKE) -C gens debug

instrumentation:
	$(MAKE) -C gens instrumentation

versions=
if PERFORMANCE_VERSION
versions+=performance
endif
if DEBUG_VERSION
versions+=debug
endif
if INSTRUMENTATION_VERSION
versions+=instrumentation
endif

check-local: $(top_srcdir)/scripts/bets
	for ver in $(versions); do \
		$(MAKE) -C gens $$ver; \
		$(top_srcdir)/scripts/bets $(BETS_OPTIONS) -o tests.log $(srcdir)/test; \
	done

dist-hook:
	cp -vr $(srcdir)/test $(distdir)
