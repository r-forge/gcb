\name{gpu.gbmv}
\alias{gpu.gbmv}
\title{On-gpu banded matrix times vector operation}

\description{
}

\usage{
gpu.gbmv(gpu.matrix, matrix.subdiagonals, matrix.superdiagonals,
	gpu.vector.x, gpu.vector.y,
	alpha=1.0, beta=0.0, matrix.ld = NULL, x.increment = 1L, y.increment = 1L,
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'))
}

\arguments{
\item{gpu.matrix}{a gpu.matrix stored in band matrix format. See Nvidia's 
	CUBLAS library documentation for the precise format.}
\item{matrix.subdiagonals}{the number of subdiagonals of gpu.matrix}
\item{matrix.superdiagonals}{the number of superdiagonals of gpu.matrix}
\item{gpu.vector.x}{a gpu.vector}
\item{gpu.vector.y}{a gpu.vector}
\item{alpha}{a scalar multiplier}
\item{beta}{a scalar multiplier}
\item{matrix.ld}{the leading dimension of gpu.matrix}
\item{x.increment}{the increment of elements of gpu.vector.x}
\item{y.increment}{the increment of elements of gpu.vector.y}
\item{matrix.transpose}{if set to 'T', 't', 'C', or 'c' use the transpose of
	gpu.matrix in the operation; otherwise, just use gpu.matrix}
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
