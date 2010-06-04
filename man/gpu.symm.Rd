\name{gpu.symm}
\alias{gpu.symm}
\title{On-gpu symmetric matrix times matrix operation}

\description{
}

\usage{
gpu.symm(gpu.matrix.a, gpu.matrix.b, gpu.matrix.c,
	alpha=1.0, beta=0.0, a.ld = NULL, b.ld = NULL, c.ld = NULL,
	a.side = c('L', 'l', 'R', 'r'), a.location = c('U', 'u', 'L', 'l'))
}

\arguments{
	\item{gpu.matrix.a}{a symmetric gpu.matrix}
	\item{gpu.matrix.b}{a gpu.matrix}
	\item{gpu.matrix.c}{a gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{beta}{a scalar multiplier}
	\item{a.ld}{the leading dimension of gpu.matrix.a}
	\item{b.ld}{the leading dimension of gpu.matrix.b}
	\item{c.ld}{the leading dimension of gpu.matrix.c}
	\item{a.side}{If set to 'L' or 'l', gpu.matrix.a is multiplied on the left
		of gpu.matrix.b. If set to 'R' or 'r', gpu.matrix.a is multiplied on the
		right of gpu.matrix.b.}
	\item{a.location}{If set to 'U' or 'u', only the uppper triangular
		portion of gpu.matrix.a will be accessed. If set to 'L' or 'l', only the
		lower diagonal portion of gpu.matrix.a will be accessed.}
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
