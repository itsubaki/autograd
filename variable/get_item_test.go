package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleGetItem() {
	// p361
	A := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	y := variable.GetItem([]int{1})(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable[1 3]([4 5 6])
	// variable[2 3]([[0 0 0] [1 1 1]])
}

func ExampleGetItem_indices() {
	// p363
	A := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)

	y := variable.GetItem([]int{0, 0, 1})(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable[3 3]([[1 2 3] [1 2 3] [4 5 6]])
	// variable[2 3]([[2 2 2] [1 1 1]])
}
