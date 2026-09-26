package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSigmoidSimple() {
	// p330
	fmt.Println(F.SigmoidSimple(variable.New(-4)))
	fmt.Println(F.SigmoidSimple(variable.New(-2)))
	fmt.Println(F.SigmoidSimple(variable.New(0.0)))
	fmt.Println(F.SigmoidSimple(variable.New(2)))
	fmt.Println(F.SigmoidSimple(variable.New(4)))

	// Output:
	// variable(0.01798621)
	// variable(0.11920292)
	// variable(0.5)
	// variable(0.880797)
	// variable(0.98201376)
}

func ExampleSigmoidSimple_backward() {
	x := variable.New(-4, 2, 0, 2, 4)

	y := F.SigmoidSimple(x)
	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[5]([0.017662708 0.104993574 0.25 0.104993574 0.017662706])
}

func ExampleSigmoidSimple_double() {
	x := variable.New(-4, 2, 0, 2, 4)

	y := F.SigmoidSimple(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[5]([0.017662708 0.104993574 0.25 0.104993574 0.017662706])
	// variable[5]([0.017027339 -0.07996249 -0 -0.07996249 -0.017027335])
}
