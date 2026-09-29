package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmax() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668424 0.017668424 0.9646632])
	// variable[2 3]([0 0 0 0 0 0])
}

func ExampleSoftmax_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.047425874 0.11920292 0.006692851 0.95257413 0.880797 0.9933072])
	// variable[2 3]([0 7.450581e-09 0 0 5.9604645e-08 0])
}

func ExampleSoftmax_axis1n() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(-1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668424 0.017668424 0.9646632])
	// variable[2 3]([0 0 0 0 0 0])
}

func ExampleSoftmax_double() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.Softmax(1)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668424 0.017668424 0.9646632])
	// variable[2 3]([0 0 0 0 0 0])
	// variable[2 3]([0 0 0 0 0 0])
}
