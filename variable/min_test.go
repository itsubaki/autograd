package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMin() {
	A := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, -5, 6},
	)

	y := variable.Min(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([[0 0 0] [0 1 0]])
}
