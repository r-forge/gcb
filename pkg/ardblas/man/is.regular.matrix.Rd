\name{is.regular.matrix}
\alias{is.regular.matrix}
\title{Error checking for non-NULL numeric matrixes}

\description{
	The is.regular.matrix function tests whether an object is a non-NULL
	numeric matrix.
}

\usage{
is.regular.matrix(m)
}

\arguments{
	\item{m}{an object to be tested}  
}

\value{
	If the input m is a non-NULL numeric matrix, then is.regular.matrix returns
	TRUE. Otherwise, is.regular.matrix returns FALSE.
}

\seealso{
is.gpu.matrix
}

\examples{
is.regular.matrix(NULL)
is.regular.matrix(matrix(c('a', 'b', 'c', 'e'), 2, 2))
is.regular.matrix(matrix(runif(4), 2, 2))
is.regular.matrix(matrix(1:4, 2, 2))
}
