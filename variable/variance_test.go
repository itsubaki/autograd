package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariance() {
	x := variable.New(
		-1, 0, 1,
		-1, 0, 1,
	).Reshape(2, 3)

	y := variable.Variance()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(-0.333, 0.333)(x.Grad))

	// Output:
	// variable(0.6666666666666666)
	// variable[2 3]([-0.333 0 0.333 -0.333 0 0.333])
}

func ExampleVariance_axis01() {
	x := variable.New(
		-1, 0, 1,
		-1, 0, 1,
	).Reshape(2, 3)

	y := variable.Variance(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(variable.Clip(-0.333, 0.333)(x.Grad))

	// Output:
	// variable(0.6666666666666666)
	// variable[2 3]([-0.333 0 0.333 -0.333 0 0.333])
}

func ExampleVariance_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Variance(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([2.25 2.25 2.25])
	// variable[2 3]([-1.5 -1.5 -1.5 1.5 1.5 1.5])
}

func ExampleVariance_axis1() {
	x := variable.New(
		-3, 0, 3,
		-3, 0, 3,
	).Reshape(2, 3)

	y := variable.Variance(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([6 6])
	// variable[2 3]([-2 0 2 -2 0 2])
}

func ExampleVariance_double() {
	x := variable.New(
		-3, 0, 3,
		-3, 0, 3,
	).Reshape(2, 3)

	y := variable.Variance(1)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad) // double backprop of Sub is zero

	// Output:
	// variable[2]([6 6])
	// variable[2 3]([-2 0 2 -2 0 2])
	// variable[2 3]([0 0 0 0 0 0])
}
