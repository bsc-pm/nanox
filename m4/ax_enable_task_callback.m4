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

