package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/matrix"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMatMul() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	w := variable.New(
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	).Reshape(3, 4)

	y := variable.MatMul(x, w)
	y.Backward()

	fmt.Println(matrix.Shape(x.Grad.Data))
	fmt.Println(matrix.Shape(w.Grad.Data))

	// Output:
	// [2 3]
	// [3 4]
}
