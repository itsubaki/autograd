package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMin() {
	A := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([[0 0 0] [0 1 0]])
}
