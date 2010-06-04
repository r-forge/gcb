\name{is.regular.vector}
\alias{is.regular.vector}
\title{Error checking for non-NULL numeric vectors}

\description{
	The is.regular.vector function tests whether an object is a non-NULL numeric
	vector.
}

\usage{
is.regular.vector(v)
}

\arguments{
	\item{v}{an object to be tested}  
}

\value{
	If the input v is a non-NULL numeric vector, then is.regular.vector returns
	TRUE. Otherwise, is.regular.vector returns FALSE.
}

\seealso{
is.gpu.vector
}

\examples{
is.regular.vector(NULL)
is.regular.vector(c('a', 'b', 'c'))
is.regular.vector(runif(5))
is.regular.vector(1:10)
}
