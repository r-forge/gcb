\name{gpu.trmv}
\alias{gpu.trmv}
\title{On-gpu triangular matrix times vector operation}

\description{
}

\usage{
gpu.trmv(gpu.matrix, gpu.vector.x, 
	matrix.ld = NULL, x.increment = 1L,
	matrix.location = c('U', 'u', 'L', 'l'),
	matrix.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	matrix.unit.tri = c('N', 'n', 'U', 'u'))
}

\arguments{
	\item{gpu.matrix}{a triangular gpu.matrix}
	\item{gpu.vector.x}{a gpu.vector}
	\item{matrix.ld}{the leading dimension of gpu.matrix}
	\item{x.increment}{the increment of elements of gpu.vector.x}
	\item{matrix.location}{If set to 'U' or 'u', then gpu.matrix is
		upper triangular. If set to 'L' or 'l', then gpu.matrix is
		lower triangular.}
	\item{matrix.transpose}{if set to 'T', 't', 'C', or 'c' use the transpose of
		gpu.matrix in the operation; otherwise, just use gpu.matrix}
	\item{matrix.unit.tri}{If set to 'U' or 'u', gpu.matrix is assumed to be 
		unit triangular. If set to 'N' or 'n', gpu.matrix is not assumed to be
		unit triangular.}
}

\value{
The result of the operation is stored in gpu.vector.x in gpu memory overwriting
the contents of gpu.vector.x. The result may be retrieved using the function
gpu.get.vector.
}

\seealso{
gpu.set.vector, gpu.set.matrix, is.increment, is.leading.dimension,
gpu.get.vector
}

\examples{
}
