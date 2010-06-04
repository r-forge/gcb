\name{gpu.trmm}
\alias{gpu.trmm}
\title{On-gpu triangular matrix times matrix operation}

\description{
}

\usage{
gpu.trmm(gpu.matrix.a, gpu.matrix.b, alpha=1.0,
	a.ld = NULL, b.ld = NULL,
	a.side = c('L', 'l', 'R', 'r'), a.location = c('U', 'u', 'L', 'l'),
	a.transpose = c('U', 'u', 'L', 'l'), a.unit.tri = c('N', 'n', 'U', 'u'))
}

\arguments{
	\item{gpu.matrix.a}{a triangular gpu.matrix}
	\item{gpu.matrix.b}{a gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{a.ld}{the leading dimension of gpu.matrix.a}
	\item{b.ld}{the leading dimension of gpu.matrix.b}
	\item{a.side}{If set to 'L' or 'l', gpu.matrix.a is multiplied on the left
		of gpu.matrix.b. If set to 'R' or 'r', gpu.matrix.a is multiplied on the
		right of gpu.matrix.b}
	\item{a.location}{If set to 'U' or 'u', then gpu.matrix.a is
		upper triangular. If set to 'L' or 'l', then gpu.matrix.a is
		lower triangular.}
	\item{a.transpose}{if set to 'T', 't', 'C', or 'c' use the transpose of
		gpu.matrix.a in the operation; otherwise, just use gpu.matrix.a}
	\item{a.unit.tri}{If set to 'U' or 'u', gpu.matrix.a is assumed to be 
		unit triangular. If set to 'N' or 'n', gpu.matrix.a is not assumed to be
		unit triangular.}
}

\value{
The result of the operation is stored in gpu.matrix.c in gpu memory overwriting
the contents of gpu.matrix.c The result may be retrieved using the function
gpu.get.matrix.
}

\seealso{
gpu.set.vector, gpu.set.matrix, is.increment, is.leading.dimension,
gpu.get.vector
}

\examples{
}
