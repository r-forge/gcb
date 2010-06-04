\name{is.gpu.vector}
\alias{is.gpu.vector}
\title{Error checking for on-gpu vectors}

\description{
	The is.gpu.vector function tests whether an object is properly formatted to
	use as an on-gpu vector. The other functions of the package use
	is.gpu.vector for error checking.
}

\usage{
is.gpu.vector(v)
}

\arguments{
	\item{v}{an object to be tested}
}

\value{
	If the input v is properly formatted for use as an on-gpu vector, then 
	is.gpu.vector returns TRUE. Otherwise, is.gpu.vector returns FALSE.
}

\examples{
candidate.vector <- NULL
is.gpu.vector(candidate.vector)
candidate.vector <- gpu.set.vector(runif(10))
is.gpu.vector(candidate.vector)
}
