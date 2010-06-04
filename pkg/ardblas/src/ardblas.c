#include<R.h>
#include<Rinternals.h>
#include<Rmath.h>
#include<cublas.h>

// cublas utilities

static char * cublasGetErrorString(cublasStatus err)
{
	switch(err) {
		case CUBLAS_STATUS_SUCCESS :
			return "operation completed successfully";
		case CUBLAS_STATUS_NOT_INITIALIZED :
			return "CUBLAS library not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED :
			return "resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE :
			return "unsupported numerical value was passed to function";
		case CUBLAS_STATUS_ARCH_MISMATCH :
			return "function requires an architectural feature absent from \
			the architecture of the device";
		case CUBLAS_STATUS_MAPPING_ERROR :
			return "access to GPU memory space failed";
		case CUBLAS_STATUS_EXECUTION_FAILED :
			return "GPU program failed to execute";
		case CUBLAS_STATUS_INTERNAL_ERROR :
			return "an internal CUBLAS operation failed";
		default :
			return "unknown error type";
	}
}

static void checkCublasError(const char * msg)
{
	cublasStatus err = cublasGetError();
	if(err != CUBLAS_STATUS_SUCCESS)
		error("cublas error : %s : %s\n", msg, cublasGetErrorString(err));
}

static int hasCublasError(const char * msg)
{
	cublasStatus err = cublasGetError();
	if(err != CUBLAS_STATUS_SUCCESS)
		error("cublas error : %s : %s\n", msg, cublasGetErrorString(err));
	return 0;
}

static void d_finalizer(SEXP d_ptr)
{
	if(!R_ExternalPtrAddr(d_ptr))
		return;
		
	cublasFree(R_ExternalPtrAddr(d_ptr));
	R_ClearExternalPtr(d_ptr);
}

// vector utilities

static SEXP packVector(int n, double * dPtr)
{
	SEXP d_ptr;
	PROTECT(d_ptr = R_MakeExternalPtr(dPtr, install("gpu vector"), R_NilValue));
	R_RegisterCFinalizerEx(d_ptr, d_finalizer, TRUE);

	SEXP vLen;
	PROTECT(vLen = allocVector(INTSXP, 1));
	INTEGER(vLen)[0] = n;

	SEXP vList, names;
	PROTECT(vList = allocVector(VECSXP, 2));
	SET_VECTOR_ELT(vList, 0, vLen);
	SET_VECTOR_ELT(vList, 1, d_ptr);

	PROTECT(names = allocVector(STRSXP, 2));
	SET_STRING_ELT(names, 0, mkChar("length"));
	SET_STRING_ELT(names, 1, mkChar("device.pointer"));
	setAttrib(vList, R_NamesSymbol, names);

	UNPROTECT(4);
	return vList;
}

// get the list element named str, or return NULL
static SEXP getListElement(SEXP list, const char *str)
{
	SEXP
		elmt = R_NilValue,
		names = getAttrib(list, R_NamesSymbol);
	int i;

	for (i = 0; i < length(list); i++) {
		if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
			elmt = VECTOR_ELT(list, i);
			break;
		}
	}
	return elmt;
}

static void unpackVector(SEXP dv, int * out_len, double ** out_ptr)
{
	SEXP len, d_ptr;

	len = getListElement(dv, "length");
	if(isNull(len) || (asInteger(len) <= 0))
		error("ardblas: unpackVector: improperly formatted gpu vector");
	*out_len = asInteger(len);

	d_ptr = getListElement(dv, "device.pointer");
	if(isNull(d_ptr) || !R_ExternalPtrAddr(d_ptr))
		error("ardblas: unpackVector: improperly formatted gpu vector");
	*out_ptr = R_ExternalPtrAddr(d_ptr);
}

SEXP advancePointer(SEXP ptr, SEXP inc)
{
	if(isNull(ptr) || !R_ExternalPtrAddr(ptr))
		error("ardblas: unpackVector: improperly formatted pointer");
	if(inc <= 0)
		error("ardblas: unpackVector: amount to advance pointer not positive");

	double * newPtr = R_ExternalPtrAddr(ptr);
	int increment = asInteger(inc);
	newPtr += increment;

	SEXP new_ptr;
	PROTECT(new_ptr = R_MakeExternalPtr(newPtr, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(new_ptr, d_finalizer, TRUE);
	UNPROTECT(1);
	return new_ptr;
}

SEXP d_setVector(SEXP v)
{
	int n = length(v);
	double * d_v;

	cublasAlloc(n, sizeof(double), (void **)&d_v);
	cublasSetVector(n, sizeof(double), REAL(v), 1, d_v, 1);
	checkCublasError("d_setVector");
	
	return packVector(n, d_v);
}

SEXP d_getVector(SEXP vList, SEXP inc)
{
	int n, increment = asInteger(inc);
	double * dPtr;
	unpackVector(vList, &n, &dPtr);

	SEXP out;
	PROTECT(out = allocVector(REALSXP, n));
	cublasGetVector(n, sizeof(double), dPtr, increment, REAL(out), 1);
	checkCublasError("d_getVector");
	UNPROTECT(1);
	return out;
}

// double precision BLAS 1 routines

SEXP d_asum(SEXP vList, SEXP inc)
{
	int n, increment = asInteger(inc);
	double * dPtr;
	unpackVector(vList, &n, &dPtr);

	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	REAL(out)[0] = cublasDasum(n, dPtr, increment);
	checkCublasError("d_asum");
	UNPROTECT(1);
	return out;
}

void d_axpy(SEXP ralpha, SEXP rx, SEXP rincx, SEXP ry, SEXP rincy)
{
	int
		nx, ny, n,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		alpha = asReal(ralpha),
		* x, * y;

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	cublasDaxpy(n, alpha, x, incx, y, incy);
	checkCublasError("d_axpy");
}

void d_copy(SEXP rx, SEXP rincx, SEXP ry, SEXP rincy)
{
	int
		nx, ny, n,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		* x, * y;

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	cublasDcopy(n, x, incx, y, incy);
	checkCublasError("d_copy");
}

SEXP d_dot(SEXP rx, SEXP rincx, SEXP ry, SEXP rincy)
{
	int
		nx, ny, n,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		* x, * y;

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	REAL(out)[0] = cublasDdot(n, x, incx, y, incy); 
	checkCublasError("d_dot");
	UNPROTECT(1);
	return out;
}

SEXP d_nrm2(SEXP rx, SEXP rincx)
{
	int
		n, 
		incx = asInteger(rincx);
	double * x;

	unpackVector(rx, &n, &x);

	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	REAL(out)[0] = cublasDnrm2(n, x, incx);
	checkCublasError("d_nrm2");
	UNPROTECT(1);
	return out;
}

void d_scal(SEXP ralpha, SEXP rx, SEXP rincx)
{
	int
		n, 
		incx = asInteger(rincx);
	double
		* x, alpha = asReal(ralpha);

	unpackVector(rx, &n, &x);

	cublasDscal(n, alpha, x, incx);
	checkCublasError("d_scal");
}

void d_swap(SEXP rx, SEXP rincx, SEXP ry, SEXP rincy)
{
	int
		nx, ny, n,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		* x, * y;

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	cublasDswap(n, x, incx, y, incy);
	checkCublasError("d_swap");
}

void d_rot(SEXP rx, SEXP rincx, SEXP ry, SEXP rincy, SEXP rsc, SEXP rss)
{
	int
		nx, ny, n,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		sc = asReal(rsc),
		ss = asReal(rss),
		* x, * y;

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	cublasDrot(n, x, incx, y, incy, sc, ss);
	checkCublasError("d_rot");
}

void d_rotm(SEXP rx, SEXP rincx, SEXP ry, SEXP rincy, SEXP rsparam)
{
	int
		nx, ny, n, ns,
		incx = asInteger(rincx),
		incy = asInteger(rincy);
	double
		* sparam,
		* x, * y;

	unpackVector(rsparam, &ns, &sparam);
	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	n = imin2(nx, ny);

	cublasDrotm(n, x, incx, y, incy, sparam);
	checkCublasError("d_rotm");
}

// matrix utilities

static SEXP packMatrix(int rows, int cols, double * dPtr)
{
	SEXP d_ptr;
	PROTECT(d_ptr = R_MakeExternalPtr(dPtr, install("gpu matrix"), R_NilValue));
	R_RegisterCFinalizerEx(d_ptr, d_finalizer, TRUE);

	SEXP rrows;
	PROTECT(rrows = allocVector(INTSXP, 1));
	INTEGER(rrows)[0] = rows;

	SEXP rcols;
	PROTECT(rcols = allocVector(INTSXP, 1));
	INTEGER(rcols)[0] = cols;

	SEXP mList, names;
	PROTECT(mList = allocVector(VECSXP, 3));
	SET_VECTOR_ELT(mList, 0, rrows);
	SET_VECTOR_ELT(mList, 1, rcols);
	SET_VECTOR_ELT(mList, 2, d_ptr);

	PROTECT(names = allocVector(STRSXP, 3));
	SET_STRING_ELT(names, 0, mkChar("rows"));
	SET_STRING_ELT(names, 1, mkChar("cols"));
	SET_STRING_ELT(names, 2, mkChar("device.pointer"));
	setAttrib(mList, R_NamesSymbol, names);

	UNPROTECT(5);
	return mList;
}

static void unpackMatrix(SEXP dm, int * out_rows, int * out_cols,
	double ** out_ptr)
{
	SEXP rows, cols, d_ptr;

	rows = getListElement(dm, "rows");
	if(isNull(rows) || (asInteger(rows) <= 0))
		error("unpackVector: gpu matrix has invalid rows");
	*out_rows = asInteger(rows);

	cols = getListElement(dm, "cols");
	if(isNull(cols) || (asInteger(cols) <= 0))
		error("ardblas: unpackVector: gpu matrix has invalid cols");
	*out_cols = asInteger(cols);

	d_ptr = getListElement(dm, "device.pointer");
	if(isNull(d_ptr) || !R_ExternalPtrAddr(d_ptr))
		error("ardblas: unpackVector: gpu matrix has invalid device pointer");
	*out_ptr = R_ExternalPtrAddr(d_ptr);
}

SEXP d_setMatrix(SEXP m)
{
	int
		rows = nrows(m), cols = ncols(m);
	double * d_m;
	
	cublasAlloc(rows * cols, sizeof(double), (void **)&d_m);
	cublasSetMatrix(rows, cols, sizeof(double), REAL(m), rows, d_m, rows);
	checkCublasError("d_setMatrix");
	
	return packMatrix(rows, cols, d_m);
}

SEXP d_getMatrix(SEXP mList, SEXP rld)
{
	int
		rows, cols, ld = asInteger(rld);
	double * dPtr;
	
	unpackMatrix(mList, &rows, &cols, &dPtr);

	SEXP out, dim;
	PROTECT(out = allocVector(REALSXP, rows * cols));
	cublasGetMatrix(rows, cols, sizeof(double), dPtr, ld, REAL(out), rows);
	checkCublasError("d_getMatrix");

	PROTECT(dim = allocVector(INTSXP, 2));
	INTEGER(dim)[0] = rows;
	INTEGER(dim)[1] = cols;
	setAttrib(out, R_DimSymbol, dim);

	UNPROTECT(2);
	return out;
}

static Rboolean isTranspose(char trans)
{
	Rboolean doit = FALSE;
	if((trans == 'T') || (trans == 't') || (trans == 'C') || (trans == 'c'))
		doit = TRUE;
	else if((trans == 'N') || (trans == 'n')) // for readers' reference
		doit = FALSE;
	return doit;
}

static char getTranspose(SEXP rtrans)
{
	char allowable[] = { 'N', 'n', 'T', 't', 'C', 'c' };
	int n = 6;
	
	for(int i = 0; i < n; i++) {
		if(CHAR(STRING_ELT(rtrans, 0))[0] == allowable[i])
			return allowable[i];
	}
	error("transpose character argument invalid");
	return 'E';
}

static char getSymLoc(SEXP ruplo)
{
	char allowable[] = { 'U', 'u', 'L', 'l' };
	int n = 4;
	
	for(int i = 0; i < n; i++) {
		if(CHAR(STRING_ELT(ruplo, 0))[0] == allowable[i])
			return allowable[i];
	}
	error("uplo character argument invalid");
	return 'E';
}

static char getUnitTri(SEXP rdiag)
{
	char allowable[] = { 'U', 'u', 'N', 'n' };
	int n = 4;
	
	for(int i = 0; i < n; i++) {
		if(CHAR(STRING_ELT(rdiag, 0))[0] == allowable[i])
			return allowable[i];
	}
	error("transpose character argument invalid");
	return 'E';
}

static char getSide(SEXP rside)
{
	char allowable[] = { 'L', 'l', 'R', 'r' };
	int n = 4;
	
	for(int i = 0; i < n; i++) {
		if(CHAR(STRING_ELT(rside, 0))[0] == allowable[i])
			return allowable[i];
	}
	error("side character argument invalid");
	return 'E';
}

// double precision BLAS 2 routines

void d_gbmv(SEXP rtrans, SEXP rkl, SEXP rku, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx, SEXP rbeta, SEXP ry, SEXP rincy)
{
	char
		trans = getTranspose(rtrans);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * x, * y;
	int
		rowsa, colsa,
		lda = asInteger(rlda),
		kl = asInteger(rkl), ku = asInteger(rku),
		nx, ny,
		incx = asInteger(rincx), incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDgbmv(trans, rowsa, colsa, kl, ku, alpha, a, lda, x, incx,
		beta, y, incy);
	checkCublasError("d_gbmv");
}

void d_gemv(SEXP rtrans, SEXP ralpha, SEXP ra, SEXP rlda, SEXP rx, SEXP rincx,
	SEXP rbeta, SEXP ry, SEXP rincy)
{
	char
		trans = getTranspose(rtrans);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * x, * y;
	int
		nx, ny, rowsa, colsa,
		lda = asInteger(rlda),
		incx = asInteger(rincx),
		incy = asInteger(rincy);
		
	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);
	
	cublasDgemv(trans, rowsa, colsa, alpha, a, lda, x, incx, beta, y, incy);
	checkCublasError("d_gemv");
}

void d_ger(SEXP ralpha, SEXP rx, SEXP rincx, SEXP ry, SEXP rincy,
	SEXP ra, SEXP rlda)
{
	double
		alpha = asReal(ralpha),
		* a, * x, * y;
	int
		rowsa, colsa,
		lda = asInteger(rlda),
		nx, ny,
		incx = asInteger(rincx),
		incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDger(rowsa, colsa, alpha, x, incx, y, incy, a, lda);
	checkCublasError("d_ger");
}

void d_sbmv(SEXP ruplo, SEXP rk, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx, SEXP rbeta, SEXP ry, SEXP rincy)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * x, * y;
	int
		rowsa, colsa, lda = asInteger(rlda), k = asInteger(rk),
		nx, ny, incx = asInteger(rincx), incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDsbmv(uplo, rowsa, k, alpha, a, lda, x, incx, beta, y, incy);
	checkCublasError("d_sbmv");
}

void d_spmv(SEXP ruplo, SEXP ralpha, SEXP ra, SEXP rx, SEXP rincx,
	SEXP rbeta, SEXP ry, SEXP rincy)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * x, * y;
	int
		rowsa, colsa,
		nx, ny, incx = asInteger(rincx), incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDspmv(uplo, rowsa, alpha, a, x, incx, beta, y, incy);
	checkCublasError("d_spmv");
}

void d_spr(SEXP ruplo, SEXP ralpha, SEXP rx, SEXP rincx, SEXP ra)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha),
		* a, * x;
	int
		rowsa, colsa,
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDspr(uplo, rowsa, alpha, x, incx, a);
	checkCublasError("d_spr");
}

void d_spr2(SEXP ruplo, SEXP ralpha, SEXP rx, SEXP rincx, SEXP ry, SEXP rincy,
	SEXP ra)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha),
		* a, * x, * y;
	int
		rowsa, colsa,
		nx, incx = asInteger(rincx),
		ny, incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDspr2(uplo, rowsa, alpha, x, incx, y, incy, a);
	checkCublasError("d_spr2");
}

void d_symv(SEXP ruplo, SEXP ralpha, SEXP ra, SEXP rlda, SEXP rx, SEXP rincx,
	SEXP rbeta, SEXP ry, SEXP rincy)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * x, * y;
	int
		nx, ny, rowsa, colsa,
		lda = asInteger(rlda),
		incx = asInteger(rincx),
		incy = asInteger(rincy);
		
	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);
	
	cublasDsymv(uplo, rowsa, alpha, a, lda, x, incx, beta, y, incy);
	checkCublasError("d_symv");
}

void d_syr(SEXP ruplo, SEXP ralpha, SEXP rx, SEXP rincx, SEXP ra, SEXP rlda)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), * a, * x;
	int
		rowsa, colsa, lda = asInteger(rlda),
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDsyr(uplo, rowsa, alpha, x, incx, a, lda);
	checkCublasError("d_syr");
}

void d_syr2(SEXP ruplo, SEXP ralpha, SEXP rx, SEXP rincx, SEXP ry, SEXP rincy,
	SEXP ra, SEXP rlda)
{
	char
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha),
		* a, * x, * y;
	int
		rowsa, colsa, lda = asInteger(rlda),
		nx, incx = asInteger(rincx),
		ny, incy = asInteger(rincy);

	unpackVector(rx, &nx, &x);
	unpackVector(ry, &ny, &y);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDsyr2(uplo, rowsa, alpha, x, incx, y, incy, a, lda);
	checkCublasError("d_syr2");
}

void d_tbmv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP rk, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans),
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa, lda = asInteger(rlda),
		k = asInteger(rk),
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtbmv(uplo, trans, diag, rowsa, k, a, lda, x, incx);
	checkCublasError("d_tbmv");
}

void d_tbsv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP rk, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans),
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa, lda = asInteger(rlda),
		k = asInteger(rk),
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtbsv(uplo, trans, diag, rowsa, k, a, lda, x, incx);
	checkCublasError("d_tbsv");
}

void d_tpmv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP ra, SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans),
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa,
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtpmv(uplo, trans, diag, rowsa, a, x, incx);
	checkCublasError("d_tpmv");
}

void d_tpsv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP ra, SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans), 
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa,
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtpsv(uplo, trans, diag, rowsa, a, x, incx);
	checkCublasError("d_tpsv");
}

void d_trmv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans), 
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa, lda = asInteger(rlda),
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtrmv(uplo, trans, diag, rowsa, a, lda, x, incx);
	checkCublasError("d_trmv");
}

void d_trsv(SEXP ruplo, SEXP rtrans, SEXP rdiag, SEXP ra, SEXP rlda,
	SEXP rx, SEXP rincx)
{
	char
		uplo = getSymLoc(ruplo),
		trans = getTranspose(rtrans), 
		diag = getUnitTri(rdiag);
	double
		* a, * x;
	int
		rowsa, colsa, lda = asInteger(rlda),
		nx, incx = asInteger(rincx);

	unpackVector(rx, &nx, &x);
	unpackMatrix(ra, &rowsa, &colsa, &a);

	cublasDtrsv(uplo, trans, diag, rowsa, a, lda, x, incx);
	checkCublasError("d_trsv");
}

// double precision BLAS 3 routines

void d_gemm(SEXP rtransa, SEXP rtransb, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rb, SEXP rldb, SEXP rbeta, SEXP rc, SEXP rldc)
{
	char
		transa = getTranspose(rtransa),
		transb = getTranspose(rtransb);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * b, * c;
	int
		m, n, k,
		rowsa, colsa, lda = asInteger(rlda),
		rowsb, colsb, ldb = asInteger(rldb),
		rowsc, colsc, ldc = asInteger(rldc);	
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rb, &rowsb, &colsb, &b);
	unpackMatrix(rc, &rowsc, &colsc, &c);
	
	m = rowsa;
	n = colsb;
	k = colsa;
	
	if(isTranspose(transa)) {
		m = colsa;
		k = rowsa;
	}
	
	if(isTranspose(transb))
		n = rowsb;
	
	cublasDgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
	checkCublasError("d_gemm");
}

void d_symm(SEXP rside, SEXP ruplo, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rb, SEXP rldb, SEXP rbeta, SEXP rc, SEXP rldc)
{
	char
		side = getSide(rside),
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * b, * c;
	int
		rowsa, colsa, lda = asInteger(rlda),
		rowsb, colsb, ldb = asInteger(rldb),
		rowsc, colsc, ldc = asInteger(rldc);	
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rb, &rowsb, &colsb, &b);
	unpackMatrix(rc, &rowsc, &colsc, &c);
	
	cublasDsymm(side, uplo, rowsb, colsb, alpha, a, lda, b, ldb,
		beta, c, ldc);
	checkCublasError("d_symm");
}

void d_syrk(SEXP ruplo, SEXP rtrans, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rbeta, SEXP rc, SEXP rldc)
{
	char
		trans = getTranspose(rtrans),
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * c;
	int
		k,
		rowsa, colsa, lda = asInteger(rlda),
		rowsc, colsc, ldc = asInteger(rldc);	
	
	k = rowsa;
	if((trans == 'N') || (trans == 'n')) {
		k = colsa;
	}
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rc, &rowsc, &colsc, &c);
	
	cublasDsyrk(uplo, trans, rowsc, k, alpha, a, lda, beta, c, ldc);
	checkCublasError("d_syrk");
}

void d_syr2k(SEXP ruplo, SEXP rtrans, SEXP ralpha, SEXP ra, SEXP rlda,
	SEXP rb, SEXP rldb, SEXP rbeta, SEXP rc, SEXP rldc)
{
	char
		trans = getTranspose(rtrans),
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha), beta = asReal(rbeta),
		* a, * b, * c;
	int
		k,
		rowsa, colsa, lda = asInteger(rlda),
		rowsb, colsb, ldb = asInteger(rldb),
		rowsc, colsc, ldc = asInteger(rldc);	
	
	k = rowsa;
	if((trans == 'N') || (trans == 'n')) {
		k = colsa;
	}
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rb, &rowsb, &colsb, &b);
	unpackMatrix(rc, &rowsc, &colsc, &c);
	
	cublasDsyr2k(uplo, trans, rowsc, k, alpha, a, lda, b, ldb, beta, c, ldc);
	checkCublasError("d_syr2k");
}

void d_trmm(SEXP rside, SEXP ruplo, SEXP rtrans, SEXP rdiag,
	SEXP ralpha, SEXP ra, SEXP rlda, SEXP rb, SEXP rldb)
{
	char
		trans = getTranspose(rtrans),
		diag = getUnitTri(rdiag),
		side = getSide(rside),
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha),
		* a, * b;
	int
		rowsa, colsa, lda = asInteger(rlda),
		rowsb, colsb, ldb = asInteger(rldb);	
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rb, &rowsb, &colsb, &b);
	
	cublasDtrmm(side, uplo, trans, diag, rowsb, colsb, alpha, a, lda, b, ldb);
	checkCublasError("d_trmm");
}

void d_trsm(SEXP rside, SEXP ruplo, SEXP rtrans, SEXP rdiag,
	SEXP ralpha, SEXP ra, SEXP rlda, SEXP rb, SEXP rldb)
{
	char
		trans = getTranspose(rtrans),
		diag = getUnitTri(rdiag),
		side = getSide(rside),
		uplo = getSymLoc(ruplo);
	double
		alpha = asReal(ralpha),
		* a, * b;
	int
		rowsa, colsa, lda = asInteger(rlda),
		rowsb, colsb, ldb = asInteger(rldb);	
		
	unpackMatrix(ra, &rowsa, &colsa, &a);
	unpackMatrix(rb, &rowsb, &colsb, &b);
	
	cublasDtrsm(side, uplo, trans, diag, rowsb, colsb, alpha, a, lda, b, ldb);
	checkCublasError("d_trsm");
}
