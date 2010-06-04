\name{is.gpu.matrix}
\alias{is.gpu.matrix}
\title{Error checking for on-gpu matrixes}

\description{
	The is.gpu.matrix function tests whether an object is properly formatted to
	use as an on-gpu matrix. The other functions of the package use
	is.gpu.matrix for error checking.
}

\usage{
is.gpu.matrix(m)
}

\arguments{
	\item{m}{an object to be tested}
}

\value{
	If the input m is properly formatted for use as an on-gpu matrix, then 
	is.gpu.matrix returns TRUE. Otherwise, is.gpu.matrix returns FALSE.
}

\examples{
candidate.matrix <- NULL
is.gpu.matrix(candidate.matrix)
candidate.matrix <- gpu.set.matrix(matrix(runif(10), 2, 5))
is.gpu.matrix(candidate.matrix)
}
