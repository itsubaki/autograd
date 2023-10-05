package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSub() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([1])
	// variable([1]) variable([-1])
}

func ExampleSubC() {
	x := variable.New(3.0)
	y := variable.SubC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([7])
	// variable([-1])
}

func ExampleSub_broadcast() {
	// p305
	a := variable.New(1, 2, 3, 4, 5)
	b := variable.New(1)
	y := variable.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([0 1 2 3 4])
	// variable([1 1 1 1 1]) variable([-5])
}
