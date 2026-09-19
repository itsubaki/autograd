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
	// variable(0.017986208)
	// variable(0.11920291)
	// variable(0.5)
	// variable(0.8807971)
	// variable(0.9820138)
}

func ExampleSigmoid_backward() {
	x := variable.New(-4, 2, 0, 2, 4)

	y := F.Sigmoid(x)
	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[5]([0.017662706 0.104993574 0.25 0.104993574 0.017662676])
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
	// variable[5]([0.017662706 0.104993574 0.25 0.104993574 0.017662676])
	// variable[5]([0.017027337 -0.07996249 0 -0.07996249 -0.017027307])
}
