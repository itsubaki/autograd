//go:build darwin

package tensor

/*
#cgo CFLAGS: -O3 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

func matmul(a, b, c []float64, m, k, n int) {
	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m),
		C.int(n),
		C.int(k),
		C.double(1.0), // alpha
		(*C.double)(unsafe.Pointer(&a[0])),
		C.int(k),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.int(n),
		C.double(0.0), // beta
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(n),
	)
}
