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

	// Output:
	// variable[1 3]([5 7 9])
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSumTo_axes21() {
	// p301
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.SumTo(2, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 1]([6 15])
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSumTo_axes11() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.SumTo(1, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSumTo_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.SumTo(1, 3)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([5 7 9])
	// variable[2 3]([1 1 1 1 1 1])
	// <nil>
}
