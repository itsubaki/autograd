//go:build !darwin

package tensor_test

import (
	"fmt"

	"github.com/itsubaki/autograd/tensor"
)

func ExampleMatMul2D() {
	a := []float64{
		1, 2,
		3, 4,
	}

	b := []float64{
		5, 6,
		7, 8,
	}

	o := make([]float64, 4)
	tensor.MatMul2D(a, b, o, 2, 2, 2)

	fmt.Printf("%.0f %.0f\n", o[0], o[1])
	fmt.Printf("%.0f %.0f\n", o[2], o[3])

	// Output:
	// 19 22
	// 43 50
}
