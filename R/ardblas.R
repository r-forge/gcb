# error checking for vectors

is.gpu.vector <- function(v)
{
	ans <- TRUE
	if(is.null(v) || !is.list(v) || is.atomic(v)) {
		ans <- FALSE
	}
	else if(is.null(v$length) || is.null(v$device.pointer)) {
		ans <- FALSE
	} else if(!is.numeric(v$length) || (v$length < 1)) {
		ans <- FALSE
	}
	return(ans)
}

is.increment <- function(i)
{
	ans <- TRUE
	if(is.null(i) || !is.numeric(i)) {
        ans <- FALSE
    } else if (i < 1) {
    	ans <- FALSE
    }
	return(ans)
}

is.regular.vector <- function(v)
{
	ans <- TRUE
	if(is.null(v) || !(is.vector(v) && is.numeric(v))) {
		ans <- FALSE
	}
	return(ans)
}

# vector utilities

gpu.set.vector <- function(v)
{
	if(!is.regular.vector(v)) {
		stop('input v must be a numeric vector')
	}
	gpu.vector <- .Call('d_setVector', v)
	return(gpu.vector)
}

gpu.get.vector <- function(gpu.vector, increment=1L)
{
	if(!is.gpu.vector(gpu.vector)) {
		stop('argument gpu.vector is not a properly formatted gpu vector')
	}
	if(!is.increment(increment)) {
		stop('argument increment must be coercible to a single integer')
	}
	
	v <- .Call('d_getVector', gpu.vector, increment)
	return(v)
}

advance.pointer <- function(ptr, increment=1L)
{
	if(is.null(ptr)) {
		stop('null ptr argument')
	}
	if(!is.increment(increment)) {
		stop('increment must be a positive integer')
	}
	new.ptr <- .Call('advancePointer', ptr, increment)
	return(new.ptr)
}

# vector ops -- BLAS 1 -- double precision

gpu.asum <- function(gpu.vector, increment=1L)
{
	if(!is.gpu.vector(gpu.vector)) {
		stop('argument gpu.vector is not a properly formatted gpu vector')
	}
	if(!is.increment(increment)) {
		stop('argument increment must be coercible to a single integer')
	}
	sum <- .Call('d_asum', gpu.vector, increment)
	return(sum)
}

gpu.axpy <- function(gpu.vector.x, gpu.vector.y, alpha=1.0,
	x.increment = 1L, y.increment = 1L)
{
	if(is.null(alpha) || !is.numeric(alpha)) {
		stop('alpha must be a real number')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both x.inc and y.inc must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_axpy', alpha, gpu.vector.x, x.increment, gpu.vector.y, y.increment)
	return()
}

gpu.copy <- function(gpu.vector.x, gpu.vector.y,
	x.increment = 1L, y.increment = 1L)
{
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_copy', gpu.vector.x, x.increment, gpu.vector.y, y.increment)
	return()
}

gpu.dot <- function(gpu.vector.x, gpu.vector.y, x.increment=1L, y.increment=1L)
{
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	dot <- .Call('d_dot', gpu.vector.x, x.increment, gpu.vector.y, y.increment)
	return(dot)
}

gpu.nrm2 <- function(gpu.vector.x, x.increment=1L)
{
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	norm <- .Call('d_nrm2', gpu.vector.x, x.increment)
	return(norm)
}

gpu.scal <- function(alpha, gpu.vector.x, x.increment=1L)
{
	if(is.null(alpha) || !is.numeric(alpha)) {
		stop('alpha must be a real number')
	}
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_scal', alpha, gpu.vector.x, x.increment)
	return()
}

gpu.swap <- function(gpu.vector.x, gpu.vector.y, x.increment=1L, y.increment=1L)
{
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_swap', gpu.vector.x, x.increment, gpu.vector.y, y.increment)
	return()
}

gpu.rot <- function(sc, ss, gpu.vector.x, gpu.vector.y, x.increment=1L,
	y.increment=1L)
{
	if(is.null(sc) || !is.numeric(sc)) {
		stop('sc must be a real number')
	}
	if(is.null(ss) || !is.numeric(ss)) {
		stop('ss must be a real number')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_rot', gpu.vector.x, x.increment, gpu.vector.y, y.increment, sc, ss)
	return()
}

gpu.rotm <- function(gpu.sparam, gpu.vector.x, gpu.vector.y, x.increment=1L,
	y.increment=1L)
{
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.sparam) && is.gpu.vector(gpu.vector.x) &&
		is.gpu.vector(gpu.vector.y))) {
		stop('x, y, sparam must be a lists each with a length and a pointer')
	}
	.Call('d_rotm', gpu.vector.x, x.increment, gpu.vector.y, y.increment,
		gpu.sparam)
	return()
}

# error checking for matrixes

is.gpu.matrix <- function(m)
{
	ans <- TRUE
	if(is.null(m) || !is.list(m) || is.atomic(m)) {
		ans <- FALSE
	} else if(is.null(m$rows) || is.null(m$cols) || is.null(m$device.pointer)) {
		ans <- FALSE
	} else if(!(is.numeric(m$rows) && is.numeric(m$cols))) {
		ans <- FALSE
	} else if((m$rows < 1) || (m$cols < 1)) {
		ans <- FALSE
	}
	return(ans)
}

is.leading.dimension <- function(i)
{
	ans <- TRUE
	if(is.null(i) || !is.numeric(i) || (i < 1)) {
        ans <- FALSE
    }
	return(ans)
}

is.regular.matrix <- function(m)
{
	ans <- TRUE
	if(is.null(m) || !(is.matrix(m) && is.numeric(m))) {
		ans <- FALSE
	}
	return(ans)
}

# matrix utilities

gpu.set.matrix <- function(m)
{
	if(!is.regular.matrix(m)) {
		stop('input m must be a numeric matrix')
	}
	gpu.matrix <- .Call('d_setMatrix', m)
	return(gpu.matrix)
}

gpu.get.matrix <- function(gpu.matrix, matrix.ld = NULL)
{
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('argument gpu.matrix is not a properly formatted gpu matrix')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('argument ld must be coercible to a single positive integer')
	}
	m <- .Call('d_getMatrix', gpu.matrix, matrix.ld)
	return(m)
}

# matrix vector ops -- BLAS 2 -- double precision

is.matrix.storage.spec <- function(spec)
{
	ans <- TRUE
	if(is.null(spec) || !is.numeric(spec) || (spec < 0)) {
		ans <- FALSE
	}
	return(ans)
}

gpu.gbmv <- function(gpu.matrix, matrix.subdiagonals, matrix.superdiagonals,
	gpu.vector.x, gpu.vector.y,
	alpha=1.0, beta=0.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'))
{
	matrix.transpose = match.arg(matrix.transpose)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	
	if(!is.matrix.storage.spec(matrix.subdiagonals)) {
		stop('subdiagonals must be coercible to a single non-negative integer')
	}
	if(!is.matrix.storage.spec(matrix.superdiagonals)) {
		stop('superdiagonals must be a single non-negative integer')
	}
	
	.Call('d_gbmv', matrix.transpose,
		matrix.subdiagonals, matrix.superdiagonals, alpha,
		gpu.matrix, matrix.ld,
		gpu.vector.x, x.increment, beta, gpu.vector.y, y.increment)
	return()
}

gpu.gemv <- function(gpu.matrix, gpu.vector.x, gpu.vector.y,
	alpha=1.0, beta=0.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'))
{
	matrix.transpose = match.arg(matrix.transpose)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_gemv', matrix.transpose, alpha, gpu.matrix, matrix.ld,
		gpu.vector.x, x.increment, beta, gpu.vector.y, y.increment)
	return()
}

gpu.ger <- function(gpu.vector.x, gpu.vector.y, gpu.matrix, alpha = 1.0,
	matrix.ld = NULL, x.increment = 1L, y.increment = 1L)
{
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	
	.Call('d_ger', alpha, gpu.vector.x, x.increment, gpu.vector.y, y.increment
		gpu.matrix, matrix.ld)
	return()
}

gpu.sbmv <- function(gpu.matrix, matrix.superdiagonals,
	gpu.vector.x, gpu.vector.y, alpha = 1.0, beta = 0.0, matrix.ld = NULL,
	x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	if(!is.matrix.storage.spec(matrix.superdiagonals)) {
		stop('superdiagonals must be a single non-negative integer')
	}
	.Call('d_sbmv', matrix.location, matrix.superdiagonals,
		alpha, gpu.matrix, matrix.ld, gpu.vector.x, x.increment,
		beta, gpu.vector.y, y.increment)
	return()
}

gpu.spmv <- function(gpu.packed.matrix, gpu.vector.x, gpu.vector.y,
	alpha = 1.0, beta = 0.0, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.packed.matrix)) {
		stop('matrix must be a list with rows, cols, and device pointer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_spmv', matrix.location, alpha, gpu.packed.matrix,
		gpu.vector.x, x.increment, beta, gpu.vector.y, y.increment)
	return()
}

gpu.spr <- function(gpu.vector.x, gpu.packed.matrix, alpha = 1.0,
	x.increment = 1L, matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.packed.matrix)) {
		stop('matrix must be a list with rows, cols, and device pointer')
	}
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_spr', matrix.location, alpha, gpu.vector.x, x.increment, 
		gpu.packed.matrix)
	return()
}

gpu.spr2 <- function(gpu.vector.x, gpu.vector.y, gpu.packed.matrix,
	alpha = 1.0, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.packed.matrix)) {
		stop('matrix must be a list with rows, cols, and device pointer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_spr2', matrix.location, alpha, gpu.vector.x, x.increment,
		gpu.vector.y, y.increment, gpu.packed.matrix)
	return()
}

gpu.symv <- function(gpu.matrix, gpu.vector.x, gpu.vector.y,
	alpha=1.0, beta=0.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_symv', matrix.location, alpha, gpu.matrix, matrix.ld,
		gpu.vector.x, x.increment, beta, gpu.vector.y, y.increment)
	return()
}

gpu.syr <- function(gpu.vector.x, gpu.matrix, alpha = 1.0, matrix.ld = NULL,
	x.increment = 1L, matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!is.increment(x.increment)) {
		stop('both x.inc must be a positive integer')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x must be a list containing a length and a pointer')
	}
	
	.Call('d_syr', matrix.location, alpha, gpu.vector.x, x.increment,
		gpu.matrix, matrix.ld)
	return()
}

gpu.syr2 <- function(gpu.vector.x, gpu.vector.y, gpu.matrix,
	alpha=1.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
{
	matrix.location = match.arg(matrix.location)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!(is.increment(x.increment) && is.increment(y.increment))) {
		stop('both incx and incy must be positive integers')
	}
	if(!(is.gpu.vector(gpu.vector.x) && is.gpu.vector(gpu.vector.y))) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	.Call('d_syr2', matrix.location, alpha, gpu.vector.x, x.increment,
		gpu.vector.y, y.increment, gpu.matrix, matrix.ld)
	return()
}

gpu.tbmv <- function(gpu.matrix, matrix.diagonals,
	gpu.vector.x, matrix.ld = NULL, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	if(!is.matrix.storage.spec(matrix.diagonals)) {
		stop('superdiagonals must be a single non-negative integer')
	}
	.Call('d_tbmv', matrix.location, matrix.transpose, matrix.unit.tri, 
		matrix.diagonals, gpu.matrix, matrix.ld, gpu.vector.x, x.increment)
	return()
}

gpu.tbsv <- function(gpu.matrix, matrix.diagonals,
	gpu.vector.x, matrix.ld = NULL, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	if(!is.matrix.storage.spec(matrix.diagonals)) {
		stop('superdiagonals must be a single non-negative integer')
	}
	.Call('d_tbsv', matrix.location, matrix.transpose, matrix.unit.tri, 
		matrix.diagonals, gpu.matrix, matrix.ld, gpu.vector.x, x.increment)
	return()
}

gpu.tpmv <- function(gpu.matrix, gpu.vector.x, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)
	
	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	
	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}
	
	.Call('d_tpmv', matrix.location, matrix.transpose, matrix.unit.tri, 
		gpu.matrix, gpu.vector.x, x.increment)
	return()
}

gpu.tpsv <- function(gpu.matrix, gpu.vector.x, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)

	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}

	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}

	.Call('d_tpsv', matrix.location, matrix.transpose, matrix.unit.tri, 
		gpu.matrix, gpu.vector.x, x.increment)
	return()
}

gpu.trmv <- function(gpu.matrix, gpu.vector.x, 
	matrix.ld = NULL, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)

	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}

	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}

	.Call('d_trmv', matrix.location, matrix.transpose, matrix.unit.tri, 
		gpu.matrix, matrix.ld, gpu.vector.x, x.increment)
	return()
}

gpu.trsv <- function(gpu.matrix, gpu.vector.x, 
	matrix.ld = NULL, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
{
	matrix.location = match.arg(matrix.location)
	matrix.transpose = match.arg(matrix.transpose)
	matrix.unit.tri = match.arg(matrix.unit.tri)

	if(!is.gpu.matrix(gpu.matrix)) {
		stop('gpu.matrix must be a list with rows, cols, and device pointer')
	}
	if(is.null(matrix.ld)) {
		matrix.ld <- gpu.matrix$rows
	}
	if(!is.leading.dimension(matrix.ld)) {
		stop('matrix.ld must be coercible to a single positive integer')
	}

	if(!is.increment(x.increment)) {
		stop('both incx and incy must be positive integers')
	}
	if(!is.gpu.vector(gpu.vector.x)) {
		stop('x and y must each be a list containing a length and a pointer')
	}

	.Call('d_trsv', matrix.location, matrix.transpose, matrix.unit.tri, 
		gpu.matrix, matrix.ld, gpu.vector.x, x.increment)
	return()
}

# matrix ops -- BLAS 3 -- double precision

gpu.gemm <- function(gpu.matrix.a, gpu.matrix.b, gpu.matrix.c,
	alpha=1.0, beta=0.0, a.ld = NULL, b.ld = NULL,
	c.ld = NULL, a.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	b.transpose = c('N', 'n', 'T', 't', 'C', 'c'))
{
	a.transpose = match.arg(a.transpose)
	b.transpose = match.arg(b.transpose)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	if(!is.gpu.matrix(gpu.matrix.c)) {
		stop('gpu.matrix.c must be a list with rows, cols, and device pointer')
	}
	if(is.null(c.ld)) {
		c.ld <- gpu.matrix.c$rows
	}
	if(!is.leading.dimension(c.ld)) {
		stop('c.ld must be coercible to a single positive integer')
	}

	.Call('d_gemm', a.transpose, b.transpose, alpha, gpu.matrix.a, a.ld, 
		gpu.matrix.b, b.ld, beta, gpu.matrix.c, c.ld)
	return()
}

gpu.symm <- function(gpu.matrix.a, gpu.matrix.b, gpu.matrix.c,
	alpha=1.0, beta=0.0, a.ld = NULL, b.ld = NULL, c.ld = NULL,
	a.side = c('L', 'l', 'R', 'r'), a.location = c('U', 'u', 'L', 'l'))
{
	a.side = match.arg(a.side)
	a.location = match.arg(a.location)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	if(!is.gpu.matrix(gpu.matrix.c)) {
		stop('gpu.matrix.c must be a list with rows, cols, and device pointer')
	}
	if(is.null(c.ld)) {
		c.ld <- gpu.matrix.c$rows
	}
	if(!is.leading.dimension(c.ld)) {
		stop('c.ld must be coercible to a single positive integer')
	}

	.Call('d_symm', a.side, a.location, alpha, gpu.matrix.a, a.ld, 
		gpu.matrix.b, b.ld, beta, gpu.matrix.c, c.ld)
	return()
}

gpu.syrk <- function(gpu.matrix.a, gpu.matrix.c, alpha=1.0, beta=0.0,
	a.ld = NULL, c.ld = NULL,
	c.location = c('U', 'u', 'L', 'l'), a.transpose = c('U', 'u', 'L', 'l'))
{
	a.side = match.arg(a.side)
	a.location = match.arg(a.location)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	if(!is.gpu.matrix(gpu.matrix.c)) {
		stop('gpu.matrix.c must be a list with rows, cols, and device pointer')
	}
	if(is.null(c.ld)) {
		c.ld <- gpu.matrix.c$rows
	}
	if(!is.leading.dimension(c.ld)) {
		stop('c.ld must be coercible to a single positive integer')
	}

	.Call('d_syrk', a.side, a.location, alpha, gpu.matrix.a, a.ld, 
		gpu.matrix.b, b.ld, beta, gpu.matrix.c, c.ld)
	return()
}

gpu.syr2k <- function(gpu.matrix.a, gpu.matrix.b, gpu.matrix.c,
	alpha=1.0, beta=0.0, a.ld = NULL, b.ld = NULL, c.ld = NULL,
	c.location = c('U', 'u', 'L', 'l'), a.transpose = c('U', 'u', 'L', 'l'))
{
	c.location = match.arg(c.location)
	a.transpose = match.arg(a.transpose)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	if(!is.gpu.matrix(gpu.matrix.c)) {
		stop('gpu.matrix.c must be a list with rows, cols, and device pointer')
	}
	if(is.null(c.ld)) {
		c.ld <- gpu.matrix.c$rows
	}
	if(!is.leading.dimension(c.ld)) {
		stop('c.ld must be coercible to a single positive integer')
	}

	.Call('d_syr2k', c.location, a.transpose, alpha, gpu.matrix.a, a.ld, 
		gpu.matrix.b, b.ld, beta, gpu.matrix.c, c.ld)
	return()
}

gpu.trmm <- function(gpu.matrix.a, gpu.matrix.b, alpha=1.0,
	a.ld = NULL, b.ld = NULL,
	a.side = c('L', 'l', 'R', 'r'), a.location = c('U', 'u', 'L', 'l'),
	a.transpose = c('U', 'u', 'L', 'l'), a.unit.tri = c('N', 'n', 'U', 'u'))
{
	a.side = match.arg(a.side)
	a.location = match.arg(a.location)
	a.transpose = match.arg(a.transpose)
	a.unit.tri = match.arg(a.unit.tri)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	.Call('d_trmm', a.side, a.location, a.transpose, a.unit.tri,
		alpha, gpu.matrix.a, a.ld, gpu.matrix.b, b.ld)
	return()
}

gpu.trsm <- function(gpu.matrix.a, gpu.matrix.b, alpha=1.0,
	a.ld = NULL, b.ld = NULL,
	a.side = c('L', 'l', 'R', 'r'), a.location = c('U', 'u', 'L', 'l'),
	a.transpose = c('U', 'u', 'L', 'l'), a.unit.tri = c('N', 'n', 'U', 'u'))
{
	a.side = match.arg(a.side)
	a.location = match.arg(a.location)
	a.transpose = match.arg(a.transpose)
	a.unit.tri = match.arg(a.unit.tri)

	if(!is.gpu.matrix(gpu.matrix.a)) {
		stop('gpu.matrix.a must be a list with rows, cols, and device pointer')
	}
	if(is.null(a.ld)) {
		a.ld <- gpu.matrix.a$rows
	}
	if(!is.leading.dimension(a.ld)) {
		stop('a.ld must be coercible to a single positive integer')
	}
	
	if(!is.gpu.matrix(gpu.matrix.b)) {
		stop('gpu.matrix.b must be a list with rows, cols, and device pointer')
	}
	if(is.null(b.ld)) {
		b.ld <- gpu.matrix.b$rows
	}
	if(!is.leading.dimension(b.ld)) {
		stop('b.ld must be coercible to a single positive integer')
	}

	.Call('d_trsm', a.side, a.location, a.transpose, a.unit.tri,
		alpha, gpu.matrix.a, a.ld, gpu.matrix.b, b.ld)
	return()
}
