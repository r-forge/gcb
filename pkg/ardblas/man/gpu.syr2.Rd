\name{gpu.syr2}
\alias{gpu.syr2}
\title{On-gpu symmetric rank 2 operation with a symmetric matrix}

\description{
}

\usage{
gpu.syr2(gpu.vector.x, gpu.vector.y, gpu.matrix,
	alpha=1.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
}

\arguments{
	\item{gpu.vector.x}{a gpu.vector}
	\item{gpu.vector.y}{a gpu.vector}
	\item{gpu.matrix}{a symmetric gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{matrix.ld}{the leading dimension of gpu.matrix}
	\item{x.increment}{the increment of elements of gpu.vector.x}
	\item{y.increment}{the increment of elements of gpu.vector.y}
	\item{matrix.location}{If set to 'U' or 'u', only the uppper triangular
		portion of gpu.matrix will be accessed. If set to 'L' or 'l', only the
		lower diagonal portion of gpu.matrix will be accessed.}
}

\value{
The result of the operation is stored in gpu.matrix in gpu memory overwriting
the contents of gpu.matrix The result may be retrieved using the function
gpu.get.matrix.
}

\seealso{
gpu.set.vector, gpu.set.matrix, is.increment, is.leading.dimension,
gpu.get.vector
}

\examples{
}
