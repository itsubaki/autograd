package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleGetItem() {
	// p361
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.GetItem([]int{1}, 0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([4 5 6])
	// variable[2 3]([0 0 0 1 1 1])
}

func ExampleGetItem_indices() {
	// p363
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.GetItem([]int{0, 0, 1}, 0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 3]([1 2 3 1 2 3 4 5 6])
	// variable[2 3]([2 2 2 1 1 1])
}

func ExampleGetItem_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
	).Reshape(2, 3)

	y := variable.GetItem([]int{1}, 0)(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([4 5 6])
	// variable[2 3]([0 0 0 1 1 1])
	// <nil>
}
