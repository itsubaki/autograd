package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMin() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMin_axis01() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-5)
	// variable[2 3]([0 0 0 0 1 0])
}

func ExampleMin_axis0() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([1 -5 3])
	// variable[2 3]([1 0 1 0 1 0])
}

func ExampleMin_axis1() {
	x := variable.New(
		1, 2, 3,
		4, -5, 6,
	).Reshape(2, 3)

	y := variable.Min(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([1 -5])
	// variable[2 3]([1 0 0 0 1 0])
}
