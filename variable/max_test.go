package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMax() {
	x := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(10)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMax_axis01() {
	x := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(10)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMax_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([4 10 6])
	// variable[2 3]([0 0 0 1 1 1])
}

func ExampleMax_axis1() {
	x := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([3 10])
	// variable[2 3]([0 0 1 0 1 0])
}

func ExampleMax_axis21n() {
	x := variable.New(
		1, 2, 3,
		4, 10, 6,
	).Reshape(2, 3)

	y := variable.Max(-2, -1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(10)
	// variable[2 3]([0 0 0 0 1 0])
}
