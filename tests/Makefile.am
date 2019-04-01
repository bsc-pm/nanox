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

CLEANFILES = tests.log tests.log.xml

EXTRA_DIST = gens/config.py \
	     lit.site.cfg \
	     litsupport/__init__.py \
	     litsupport/bets.py

if LIT_SUPPORT
check-local: lit-check
else
check-local: bets-check
endif

LIT_OPTIONS = --xunit-xml-output=tests.log.xml -sv --threads=$(shell nproc)
LIT = PYTHONPATH=$(top_srcdir)/tests $(top_srcdir)/scripts/lit/lit.py $(LIT_OPTIONS)

lit-check:
	$(LIT) $(srcdir)/test

bets-check: $(top_srcdir)/scripts/bets
	$(top_srcdir)/scripts/bets $(BETS_OPTIONS) -o tests.log $(srcdir)/test

dist-hook:
	cp -vr $(srcdir)/test $(distdir)
