\name{gpu.syrk}
\alias{gpu.syrk}
\title{On-gpu symmetric rank k operation}

\description{
}

\usage{
gpu.syrk(gpu.matrix.a, gpu.matrix.c, alpha=1.0, beta=0.0,
	a.ld = NULL, c.ld = NULL,
	c.location = c('U', 'u', 'L', 'l'), a.transpose = c('U', 'u', 'L', 'l'))
}

\arguments{
	\item{gpu.matrix.a}{a gpu.matrix}
	\item{gpu.matrix.c}{a symmetric gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{beta}{a scalar multiplier}
	\item{a.ld}{the leading dimension of gpu.matrix.a}
	\item{c.ld}{the leading dimension of gpu.matrix.c}
	\item{c.location}{If set to 'U' or 'u', only the uppper triangular
		portion of gpu.matrix.c will be accessed. If set to 'L' or 'l', only the
		lower diagonal portion of gpu.matrix.c will be accessed.}
	\item{a.transpose}{If set to 'N' or 'n', the transpose of gpu.matrix.a is
		multiplied on the right of gpu.matrix.a. If set to 'T', 't', 'C' or 'c'
		the transpose is multiplied on the left of gpu.matrix.a.}
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
