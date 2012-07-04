SUBROUTINE nanos_reduction_default_cleanup_fortran_aux(P)
    INTEGER, POINTER :: P(:)

    INTEGER :: S
    DEALLOCATE(P, STAT=S)
    IF (S /= 0) THEN
        CALL nanos_handle_error(1) ! NANOS_UNKOWN_ERROR
    END IF
END SUBROUTINE nanos_reduction_default_cleanup_fortran_aux


