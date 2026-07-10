//go:build darwin && arm64

package tensor

/*
#cgo CFLAGS: -O3 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

func matmul(a, b, c []float64, m, k, n int) {
	alpha := C.double(1.0)
	beta := C.double(0.0)

	C.cblas_dgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m),
		C.int(n),
		C.int(k),
		alpha,
		(*C.double)(unsafe.Pointer(&a[0])),
		C.int(k),
		(*C.double)(unsafe.Pointer(&b[0])),
		C.int(n),
		beta,
		(*C.double)(unsafe.Pointer(&c[0])),
		C.int(n),
	)
}
