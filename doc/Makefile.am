doxygen : doxy-nanox
	@echo "-- Running doxygen"
	doxygen doxy-nanox

doxygen-pdf : doxygen
	cd latex && make all

clean-local:
	-rm -fr html latex

