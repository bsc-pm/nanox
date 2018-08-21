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

#
# SYNOPSIS
#
#   AX_ENABLE_TASK_CALLBACK
#
# DESCRIPTION
#
#   Check whether to enable task callback feature
#

AC_DEFUN([AX_ENABLE_TASK_CALLBACK],
[
   AC_ARG_ENABLE([task-callback],
      AS_HELP_STRING([--enable-task-callback], [Enables task callback feature (disabled by default)]),
      [
         AS_IF([test "$enableval" = yes],[AC_DEFINE([NANOX_TASK_CALLBACK],[],[Enables task callback feature])])
      ]
   )

]
)dnl AX_ENABLE_TASK_CALLBACK

