package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSigmoid() {
	// p330
	fmt.Println(F.Sigmoid(variable.New(-4)))
	fmt.Println(F.Sigmoid(variable.New(-2)))
	fmt.Println(F.Sigmoid(variable.New(0.0)))
	fmt.Println(F.Sigmoid(variable.New(2)))
	fmt.Println(F.Sigmoid(variable.New(4)))

	// Output:
	// variable(0.01798620996209155)
	// variable(0.11920292202211757)
	// variable(0.5)
	// variable(0.8807970779778824)
	// variable(0.9820137900379085)
}

func ExampleSigmoid_backward() {
	x := variable.New(-4, 2, 0, 2, 4)

	y := F.Sigmoid(x)
	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[5]([0.017662706213291107 0.10499358540350653 0.25 0.10499358540350653 0.017662706213291107])
}

func ExampleSigmoid_double() {
	x := variable.New(-4, 2, 0, 2, 4)

	y := F.Sigmoid(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[5]([0.017662706213291107 0.10499358540350653 0.25 0.10499358540350653 0.017662706213291107])
	// variable[5]([0.01702733592838912 -0.07996250105615307 0 -0.07996250105615307 -0.01702733592838912])
}
