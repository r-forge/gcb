\name{gpu.ger}
\alias{gpu.ger}
\title{On-gpu symmetric rank 1 operation}

\description{
}

\usage{
gpu.ger(gpu.vector.x, gpu.vector.y, gpu.matrix, alpha = 1.0,
	matrix.ld = NULL, x.increment = 1L, y.increment = 1L)
}

\arguments{
	\item{gpu.vector.x}{a gpu.vector}
	\item{gpu.vector.y}{a gpu.vector}
	\item{gpu.matrix}{a gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{matrix.ld}{the leading dimension of gpu.matrix}
	\item{x.increment}{the increment of elements of gpu.vector.x}
	\item{y.increment}{the increment of elements of gpu.vector.y}
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
