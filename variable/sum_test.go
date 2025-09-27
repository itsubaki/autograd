package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSum() {
	// p292
	x := variable.New(1, 2, 3, 4, 5, 6)
	y := variable.Sum()(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[6]([1 1 1 1 1 1])
}

func ExampleSum_axis01() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Sum(0, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSum_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Sum(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([5 7 9])
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSum_axis1() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Sum(1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2]([6 15])
	// variable[2 3]([1 1 1 1 1 1])
}

func ExampleSum_axis21n() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.Sum(-2, -1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(21)
	// variable[2 3]([1 1 1 1 1 1])
}
