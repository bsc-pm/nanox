
include $(top_srcdir)/src/common.am

bin_PROGRAMS = nanox-list-plugins


nanox_list_plugins_CPPFLAGS = $(common_includes) $(common_cxxflags) 

nanox_list_plugins_SOURCES=\
	list_plugins.cpp \
	$(END)

nanox_list_plugins_LDADD=\
	-ldl \
	../core/libnanox.la \
	$(END)

