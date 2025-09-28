package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSplit() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := variable.Split([]int{1, 2}, 0)(x)
	fmt.Println(y[0])
	fmt.Println(y[1])

	y[0].Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([1 2 3])
	// variable[2 3]([4 5 6 7 8 9])
	// variable[3 3]([1 1 1 0 0 0 0 0 0])
}

func ExampleSplit_y1() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := variable.Split([]int{1, 2}, 0)(x)
	fmt.Println(y[0])
	fmt.Println(y[1])

	y[1].Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([1 2 3])
	// variable[2 3]([4 5 6 7 8 9])
	// variable[3 3]([0 0 0 1 1 1 1 1 1])
}

func ExampleSplit_y01() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := variable.Split([]int{1, 2}, 0)(x)
	fmt.Println(y[0])
	fmt.Println(y[1])

	y[0].Backward()
	y[1].Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([1 2 3])
	// variable[2 3]([4 5 6 7 8 9])
	// variable[3 3]([1 1 1 1 1 1 1 1 1])
}

func ExampleSplit_axis1() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := variable.Split([]int{1, 2}, 1)(x)
	fmt.Println(y[0])
	fmt.Println(y[1])

	y[0].Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[3 1]([1 4 7])
	// variable[3 2]([2 3 5 6 8 9])
	// variable[3 3]([1 0 0 1 0 0 1 0 0])
}

func ExampleSplit_double() {
	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := variable.Split([]int{1, 2}, 0)(x)
	fmt.Println(y[0])
	fmt.Println(y[1])

	y[0].Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(gx.Grad)

	// Output:
	// variable[1 3]([1 2 3])
	// variable[2 3]([4 5 6 7 8 9])
	// variable[3 3]([1 1 1 0 0 0 0 0 0])
	// <nil>
}
