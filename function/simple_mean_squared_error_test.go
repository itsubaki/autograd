package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMeanSquaredErrorSimple() {
	x0 := variable.New(
		1, 2, 3,
		1, 2, 3,
	).Reshape(2, 3)

	x1 := variable.New(
		3, 4, 5,
		3, 4, 5,
	).Reshape(2, 3)

	y := F.MeanSquaredErrorSimple(x0, x1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable(12)
	// variable[2 3]([-2 -2 -2 -2 -2 -2])
	// variable[2 3]([2 2 2 2 2 2])
}
