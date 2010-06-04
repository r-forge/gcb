\name{is.matrix.storage.spec}
\alias{is.matrix.storage.spec}
\title{Error checking for matrix storage specifier arguments}

\description{
	The is.matrix.storage.spec function tests whether an object may be used
	to specify a certain number of rows or diagonals of the matrix. For
	example, an object may be used as the number of superdiagonals of a
	matrix. This function tests whether the object is suitable for that task.
	The other functions of the package use is.matrix.storage.spec for error
	checking.
}

\usage{
is.matrix.storage.spec(i)
}

\arguments{
	\item{i}{an object to be tested}
}

\value{
	If the input i is properly formatted for use as a matrix storage specifier,
	is.matrix.storage.spec evaluates to TRUE. Otherwise,
	is.matrix.storage.spec evaluates to FALSE.
}

\examples{
	is.matrix.storage.spec(NULL)
	is.matrix.storage.spec(NA)
	is.matrix.storage.spec(100L)
	is.matrix.storage.spec(3.14)
}
