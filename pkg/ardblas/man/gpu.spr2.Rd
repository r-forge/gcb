\name{gpu.spr2}
\alias{gpu.spr2}
\title{On-gpu symmetric rank 2 operation with a packed symmetric matrix}

\description{
}

\usage{
gpu.spr2(gpu.vector.x, gpu.vector.y, gpu.packed.matrix,
	alpha = 1.0, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
}

\arguments{
	\item{gpu.vector.x}{a gpu.vector}
	\item{gpu.vector.y}{a gpu.vector}
	\item{gpu.packed.matrix}{a symmetric gpu.matrix stored in
		packed matrix format. See Nvidia's CUBLAS library documentation for
		the precise format.}
	\item{alpha}{a scalar multiplier}
	\item{x.increment}{the increment of elements of gpu.vector.x}
	\item{y.increment}{the increment of elements of gpu.vector.y}
	\item{matrix.location}{If set to 'U' or 'u', then the upper triangular
		part of the matrix is stored in gpu.matrix in packed format. If set to 
		'L' or 'l', then the lower triangular part of the matrix is stored in
		gpu.matrix in packed format. See Nvidia's CUBLAS library documentation
		for the precise format.}
}

\value{
The result of the operation is stored in gpu.packed.matrix in gpu memory
overwriting the contents of gpu.packed.matrix The result may be retrieved
using the function gpu.get.matrix.
}

\seealso{
gpu.set.vector, gpu.set.matrix, is.increment, is.leading.dimension,
gpu.get.vector
}

\examples{
}
