AC_INIT([gputools], 0.01)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([cuda-home],
	AC_HELP_STRING([--with-cuda-home=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]),
	[CUDA_HOME="$withval"])

AC_ARG_WITH([cula-home],
	AC_HELP_STRING([--with-cula-home=cula_home],
		[specify the root of your cula install, e.g. /usr/local/cula]),
	[CUDA_HOME="$withval"])

AC_ARG_WITH([r-include],
	AC_HELP_STRING([--with-r-include=r_include],
		[specify the full path to your R header files, e.g. /usr/share/R/include]),
	[R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
	AC_HELP_STRING([--with-r-lib=r-lib],
		[specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
	[R_LIB="$withval"])

device_emu=""
AC_ARG_ENABLE([emulation],
	AC_HELP_STRING([--enable-emulation],
		[emulate the gpu because maybe you don't have one]),
	[device_emu=$enableval])

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
	AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
	CUDA_HOME="/usr/local/cuda"
else
	AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_MSG_CHECKING("environment variable CULA_HOME")
if test -z "${CULA_HOME}"; then
	AC_MSG_RESULT("CULA_HOME not set; using default /usr/local/cula")
	CULA_HOME="/usr/local/cula"
else
	AC_MSG_RESULT("using CULA_HOME=${CULA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
	AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
	AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
	AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
	CUDA_LIB_DIR="/lib"
else
	AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
	CUDA_LIB_DIR="/lib64"
fi

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
	AC_MSG_ERROR("could not determine R_HOME")
else
	CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
	if test -z "${R_INCLUDE}"; then
		R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
	else
		R_INCLUDE="-I${R_INCLUDE}"
	fi
	if test -z "${R_LIB}"; then
		R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
	else
		R_LIB="-L${R_LIB} -lR"
	fi
	AC_MSG_RESULT("using ${R_INCLUDE} for R header files")
	AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CULA")
AC_CHECK_FILE([${CULA_HOME}/lib64/libcula.so], [CULA64BIT="yes"])
if test -z "${CULA64BIT}"; then
	AC_MSG_RESULT("no")
	AC_MSG_CHECKING("checking whether cula is present at all")
	AC_CHECK_FILE([${CULA_HOME}/lib/libcula.so], [HAS_CULA="yes"])
	if test -z "${HAS_CULA}"; then
		AC_MSG_RESULT("cula is not installed; gpuFastICA is disabled")
		HAS_CULA="no"
		CULA_OBJS=""
		LIBS="-L${CUDA_HOME}${CUDA_LIB_DIR} -lcublas -lcuda"
	else
		AC_MSG_RESULT("yes -- using ${CULA_HOME}/lib for CULA libs")
		CULA_LIB_DIR="/lib"
		CULA_OBJS="ica.o svd.o"
		LIBS="-L${CULA_HOME}${CULA_LIB_DIR} -lcula -lcublas -lcuda"
	fi
else
	AC_MSG_RESULT("yes -- using ${CULA_HOME}/lib64 for CULA libs")
	HAS_CULA="yes"
	CULA_LIB_DIR="/lib64"
	CULA_OBJS="ica.o svd.o"
	BLAS_LIBS=`"${R_HOME}/bin/R" CMD config BLAS_LIBS`
	LIBS="-L${CULA_HOME}${CULA_LIB_DIR} -lcula -lcublas -lcuda"
fi

if test "${device_emu}" = yes ; then
	EMULATION="-deviceemu"
	DEVICEOPTS=""
	CULA_OBJS=""
	CULA_LIB_DIR=""
	OBJS13="kendall.o"
	SVMDISABLED="TRUE"
	LIBS="-L${CUDA_HOME}${CUDA_LIB_DIR} -lcublasemu"
else
	EMULATION=""
	DEVICEOPTS="-arch=sm_13"
	OBJS13="kendall.o classification.o"
	SVMDISABLED="FALSE"
fi

RPATHFLAG=""
AC_CANONICAL_HOST
case "${host_os}" in
	*darwin*)
		if test -z "${CULA_LIB_DIR}"; then
			RPATHFLAG="-Wl,-rpath,${CUDA_HOME}${CUDA_LIB_DIR}"
		else
			RPATHFLAG="-Wl,-rpath,${CULA_HOME}${CULA_LIB_DIR}"
		fi
	;;
esac

AC_SUBST(LIBS)
AC_SUBST(CUDA_HOME)

AC_SUBST(HAS_CULA)
AC_SUBST(CULA_HOME)
AC_SUBST(CULA_OBJS)

AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)

AC_SUBST(CPICFLAGS)
AC_SUBST(EMULATION)
AC_SUBST(DEVICEOPTS)
AC_SUBST(OBJS13)
AC_SUBST(SVMDISABLED)
AC_SUBST(RPATHFLAG)

AC_CONFIG_FILES([src/Makefile])
AC_CONFIG_FILES([R/gpuSvm.R])
AC_CONFIG_FILES([R/gpuFastICA.R])
AC_CONFIG_FILES([R/gpuSvd.R])
AC_OUTPUT
