//go:build darwin

package tensor

/*
#cgo CFLAGS: -O3 -DACCELERATE_NEW_LAPACK
#cgo LDFLAGS: -framework Accelerate
#include <Accelerate/Accelerate.h>
*/
import "C"

import "unsafe"

func matmul(a, b, c []float32, m, k, n int) {
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		C.int(m),
		C.int(n),
		C.int(k),
		C.float(1.0), // alpha
		(*C.float)(unsafe.Pointer(&a[0])),
		C.int(k),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.int(n),
		C.float(0.0), // beta
		(*C.float)(unsafe.Pointer(&c[0])),
		C.int(n),
	)
}
