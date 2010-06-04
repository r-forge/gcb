\name{is.leading.dimension}
\alias{is.leading.dimension}
\title{Error checking for matrix.ld arguments}

\description{
	The is.leading.dimension function tests whether an object may be used as a
	matrix.ld for a matrix. The matrix.ld defines the space needed to get to the
	next row in matrix element units. For example, an leading.dimension of 5L
	means 'the next row is 5 elements from the beginning of the current row'.
	You can set the matrix.ld to something larger than the number of rows to
	operate on a submatrix. The other functions of the package use
	is.leading.dimension for error checking.
}

\usage{
is.leading.dimension(i)
}

\arguments{
	\item{i}{an object to be tested}
}

\value{
	If the input i is properly formatted for use as a matrix.ld for any
	matrix, is.leading.dimension evaluates to TRUE. Otherwise,
	is.leading.dimension evaluates to FALSE.
}

\examples{
	is.leading.dimension(NULL)
	is.leading.dimension(NA)
	is.leading.dimension(100L)
	is.leading.dimension(3.14)
}
