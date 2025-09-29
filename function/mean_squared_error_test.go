package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMeanSquaredError() {
	x0 := variable.New(
		1, 1, 1, 1,
		1, 1, 1, 1,
	).Reshape(2, 4)

	x1 := variable.New(
		3, 3, 3, 3,
		3, 3, 3, 3,
	).Reshape(2, 4)

	y := F.MeanSquaredError(x0, x1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable(4)
	// variable[2 4]([-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5])
	// variable[2 4]([0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5])
}

func ExampleMeanSquaredError_double() {
	x0 := variable.New(
		1, 1, 1, 1,
		1, 1, 1, 1,
	).Reshape(2, 4)

	x1 := variable.New(
		3, 3, 3, 3,
		3, 3, 3, 3,
	).Reshape(2, 4)

	y := F.MeanSquaredError(x0, x1)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	gx0 := x0.Grad
	gx1 := x1.Grad
	x0.Cleargrad()
	x1.Cleargrad()

	gx0.Backward()
	gx1.Backward()
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable(4)
	// variable[2 4]([-0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5])
	// variable[2 4]([0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5])
	// variable[2 4]([0 0 0 0 0 0 0 0])
	// variable[2 4]([0 0 0 0 0 0 0 0])
}
