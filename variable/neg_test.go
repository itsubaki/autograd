package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleNeg() {
	// p139
	x := variable.New(3.0)
	y := variable.Neg(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(-3)
	// variable(-1)
}

func ExampleNeg_double() {
	x := variable.New(3.0)

	y := variable.Neg(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable(-3)
	// variable(-1)
	// <nil>
}
