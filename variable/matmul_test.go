package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMatMul() {
	x := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	w := variable.NewOf(
		[]float64{1, 2, 3, 4},
		[]float64{5, 6, 7, 8},
		[]float64{9, 10, 11, 12},
	)

	y := variable.MatMul(x, w)
	y.Backward()

	fmt.Println(matrix.Shape(x.Grad.Data))
	fmt.Println(matrix.Shape(w.Grad.Data))

	// Output:
	// [2 3]
	// [3 4]
}
