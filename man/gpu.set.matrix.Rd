\name{gpu.set.matrix}
\alias{gpu.set.matrix}
\title{Copy a matrix to gpu memory}

\description{
	The gpu.set.matrix function copies a matrix to gpu memory. You may use the
	returned object as an argument to the ardblas library's linear algebra
	routines. When you're finished manipulating the matrix, you may retrieve
	the entries of the matrix by calling gpu.get.matrix.
	
	This function is possibly very time consuming: Gpu memory is high bandwith
	but very high latency!
}

\usage{
gpu.set.matrix(m)
}

\arguments{
	\item{m}{a non-NULL numeric matrix that the user wishes to copy to gpu
		memory. Column and row names are not preserved so keep track of them
		yourself if you want them.}  
}

\value{
	After copying the matrix to gpu memory, the function returns a list
	containing rows, cols, and device.pointer. The rows and cols entries hold
	the number of rows and number of columns of the matrix respectively. The
	rows, cols, and device.pointer elements
	may be manipulated by the user to achieve various effects. To err on the
	side of caution, please use the advance.pointer function to manipulate
	the device.pointer element. The list returned is ment to be used as an
	argument to the linear algebra routines. Once the user is finished with the
	matrix, the user may get the possibly modified matrix by calling
	gpu.get.matrix.
}

\seealso{
gpu.get.matrix, advance.pointer, is.regular.matrix
}

\examples{
gpu.matrix <- gpu.set.matrix(matrix(runif(10), 2, 5))
regular.matrix <- gpu.get.matrix(gpu.matrix)
}
