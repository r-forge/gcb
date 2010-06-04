\name{is.increment}
\alias{is.increment}
\title{Error checking for vector increment arguments}

\description{
	The is.increment function tests whether an object may be used as an
	increment for a vector. The increment defines the spacing of vector
	elements in memory. For example, an increment of 2L means 'use every
	other memory slot' as the elements of a vector. The other functions of the
	package use is.increment for error checking.
}

\usage{
is.increment(i)
}

\arguments{
	\item{i}{an object to be tested}
}

\value{
	If the input i is properly formatted for use as an increment for any
	vector's elements, is.increment evaluates to TRUE. Otherwise, is.increment
	evaluates to FALSE.
}

\examples{
	is.increment(NULL)
	is.increment(NA)
	is.increment(1L)
	is.increment(3.14)
}
