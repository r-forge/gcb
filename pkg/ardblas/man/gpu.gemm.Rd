\name{gpu.gemm}
\alias{gpu.gemm}
\title{On-gpu matrix times matrix operation}

\description{
}

\usage{
gpu.gemm(gpu.matrix.a, gpu.matrix.b, gpu.matrix.c,
	alpha=1.0, beta=0.0, a.ld = NULL, b.ld = NULL,
	c.ld = NULL, a.transpose = c('N', 'n', 'T', 't', 'C', 'c'),
	b.transpose = c('N', 'n', 'T', 't', 'C', 'c'))
}

\arguments{
	\item{gpu.matrix.a}{a gpu.matrix}
	\item{gpu.matrix.b}{a gpu.matrix}
	\item{gpu.matrix.c}{a gpu.matrix}
	\item{alpha}{a scalar multiplier}
	\item{beta}{a scalar multiplier}
	\item{a.ld}{the leading dimension of gpu.matrix.a}
	\item{b.ld}{the leading dimension of gpu.matrix.b}
	\item{c.ld}{the leading dimension of gpu.matrix.c}
	\item{a.transpose}{if set to 'T', 't', 'C', or 'c' use the transpose of
		gpu.matrix.a in the operation; if set to 'N' or 'n', just use 
		gpu.matrix.a}
	\item{b.transpose}{if set to 'T', 't', 'C', or 'c' use the transpose of
		gpu.matrix.b in the operation; if set to 'N' or 'n', just use 
		gpu.matrix.b}
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
