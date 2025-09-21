package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMax() {
	A := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max(A)
	y.Backward()

	fmt.Println(y)
	fmt.Println(A.Grad)

	// Output:
	// variable(10)
	// variable[2 3]([[0 0 0] [0 1 0]])
}
