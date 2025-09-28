package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMean() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.1666)(x.Grad))

	// Output:
	// variable(3.5)
	// variable[2 3]([0.1666 0.1666 0.1666 0.1666 0.1666 0.1666])
}

func ExampleMean_axis01() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.1666)(x.Grad))

	// Output:
	// variable(3.5)
	// variable[2 3]([0.1666 0.1666 0.1666 0.1666 0.1666 0.1666])
}

func ExampleMean_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([2.5 3.5 4.5])
	// variable[2 3]([0.5 0.5 0.5 0.5 0.5 0.5])
}

func ExampleMean_axis1() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(0, 0.3333)(x.Grad))

	// Output:
	// variable[2]([2 5])
	// variable[2 3]([0.3333 0.3333 0.3333 0.3333 0.3333 0.3333])
}

func ExampleMean_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Mean(0, 1)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(3.5)
	// <nil>
}
