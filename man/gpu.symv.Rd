\name{gpu.symv}
\alias{gpu.symv}
\title{On-gpu symmetric matrix times vector operation}

\description{
}

\usage{
gpu.symv(gpu.matrix, gpu.vector.x, gpu.vector.y, alpha=1.0, beta=0.0,
	matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'))
}

\arguments{
	\item{gpu.matrix}{a symmetric gpu.matrix}
	\item{gpu.vector.x}{a gpu.vector}
	\item{gpu.vector.y}{a gpu.vector}
	\item{alpha}{a scalar multiplier}
	\item{beta}{a scalar multiplier}
	\item{matrix.ld}{the leading dimension of gpu.matrix}
	\item{x.increment}{the increment of elements of gpu.vector.x}
	\item{y.increment}{the increment of elements of gpu.vector.y}
	\item{matrix.location}{If set to 'U' or 'u', only the uppper triangular
		portion of gpu.matrix will be accessed. If set to 'L' or 'l', only the
		lower diagonal portion of gpu.matrix will be accessed.}
}

\value{
The result of the operation is stored in gpu.vector.y in gpu memory overwriting
the contents of gpu.vector.y. The result may be retrieved using the function
gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.set.matrix, is.increment, is.leading.dimension,
gpu.get.vector
}

\examples{
}
