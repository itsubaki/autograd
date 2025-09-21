package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleTranspose() {
	// p286
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Transpose(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 2]([[1 4] [2 5] [3 6]])
	// variable[2 3]([[1 1 1] [1 1 1]])
}
