\name{advance.pointer}
\alias{advance.pointer}
\title{Manipulate the device.pointer element of a gpu.vector list}

\description{
	The advance.pointer function uses C pointer arithmetic to add to a
	device.pointer element of a gpu.vector list (made by the gpu.set.vector
	function). You can use the function to add an offset to the beginning of
	a vector or a matrix in gpu memory. For exmaple, you may want to skip the
	first element of a vector or the first 5 rows of a matrix in gpu memory.
	
	In conjunction with increment and ld arguments, this will allow you to
	apply various transforms to subvectors and submatrices!
}

\usage{
advance.pointer(ptr, increment=1L)
}

\arguments{
	\item{ptr}{
		A pointer addressing gpu memory. This will most likely come from the
		device.pointer element of the list returned by gpu.set.vector.
	}
	\item{increment}{
		The amount that you want to add to the device pointer. This should be
		in units of vector/matrix elements (not in bytes or bits).
	}
}

\value{
	The advance.pointer function returns a new device.pointer addressing
	gpu memory at the position ptr + increment.
}

\seealso{
	gpu.set.vector, gpu.get.vector, is.gpu.vector
}

\examples{
# retrieve only those elements of a vector with even indices
gpu.vector <- set.vector(runif(10))
new.vector <- gpu.vector
new.vector$device.pointer <- advance.pointer(gpu.vector$device.pointer, 1L)
new.vector$length <- 5L
evens <- gpu.get.vector(new.vector)

# skip the first row of a matrix
host.matrix <- matrix(runif(15), 3, 5)
gpu.matrix <- gpu.set.matrix(host.matrix)
short.gpu.matrix <- gpu.matrix
to.add <- short.gpu.matrix$rows
short.gpu.matrix$device.pointer <- advance.pointer(gpu.matrix$device.pointer,
	to.add)
short.gpu.matrix$rows <- gpu.matrix$rows - 1L
short.host.matrix <- gpu.get.matrix(short.gpu.matrix)
}
