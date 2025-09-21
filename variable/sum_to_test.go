package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSumTo() {
	// p301
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.SumTo(1, 3)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	x.Cleargrad()

	y = variable.SumTo(2, 1)(x)
	y.Backward()
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([5 7 9])
	// variable[2 3]([[1 1 1] [1 1 1]])
	// variable[2 1]([[6] [15]])
	// variable[2 3]([[1 1 1] [1 1 1]])
}
