package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleDiv() {
	a := variable.New(10)
	b := variable.New(2)
	y := variable.Div(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([5])
	// variable([0.5]) variable([-2.5])
}

func ExampleDivC() {
	a := 10.0
	b := variable.New(2)
	y := variable.DivC(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(b.Grad)

	// Output:
	// variable([5])
	// variable([-2.5])
}

func ExampleDiv_broadcast() {
	// p305
	a := variable.New(1, 2, 3, 4, 5)
	b := variable.New(2)
	y := variable.Div(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([0.5 1 1.5 2 2.5])
	// variable([0.5 0.5 0.5 0.5 0.5]) variable([-3.75])
}
