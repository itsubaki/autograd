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
	// variable[5]
	// variable[0.5] variable[-2.5]
}

func ExampleDiv_double() {
	a := variable.New(10)
	b := variable.New(2)
	y := variable.Div(a, b)
	y.Backward()

	ga, gb := a.Grad, b.Grad
	a.Cleargrad()
	b.Cleargrad()
	ga.Backward()
	gb.Backward()
	fmt.Println(a.Grad, b.Grad)
	fmt.Println(y.Grad.Grad)

	// Output:
	// variable[-0.25] variable[2.25]
	// variable[-2]
}
