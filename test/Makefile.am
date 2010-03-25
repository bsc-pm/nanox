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

check-local: $(top_srcdir)/scripts/bets
	$(top_srcdir)/scripts/bets -o tests.log $(srcdir)/test

dist-hook:
	mkdir $(distdir)/test; \
	for i in $(srcdir)/test/*; \
	do  \
	   DIR=test/$$(basename $$i); \
	   [ -f $$i ] && continue; \
	   [ "$DIR" = "gens" ] && continue;\
	   mkdir -p $(distdir)/$${DIR} ; \
	   find $(srcdir)/$${DIR} "(" -name "*.c" -o -name "*.cpp" -o -name "*.C" -o -name "*.cc" ")" -exec cp {} $(distdir)/$${DIR} ';' ; \
	done

