package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleConcat() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.New(
		7, 8,
		9, 10,
	).Reshape(2, 2)

	axis := 1
	z := variable.Concat(axis)(x, y)
	z.Backward()

	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable[2 5]([1 2 3 7 8 4 5 6 9 10])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[2 2]([1 1 1 1])
}

func ExampleConcat_axis0() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.New(
		7, 8, 9,
	).Reshape(1, 3)

	axis := 0
	z := variable.Concat(axis)(x, y)
	z.Backward()

	fmt.Println(z)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable[3 3]([1 2 3 4 5 6 7 8 9])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[1 3]([1 1 1])
}
