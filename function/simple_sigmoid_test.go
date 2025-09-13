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
	// variable(0.01798620996209156)
	// variable(0.11920292202211755)
	// variable(0.5)
	// variable(0.8807970779778823)
	// variable(0.9820137900379085)
}

func ExampleSigmoidSimple_backward() {
	for _, v := range []float64{-4, 2, 0, 2, 4} {
		x := variable.New(v)
		y := F.SigmoidSimple(x)
		y.Backward()

		fmt.Println(x.Grad)
	}

	// Output:
	// variable(0.017662706213291118)
	// variable(0.1049935854035065)
	// variable(0.25)
	// variable(0.1049935854035065)
	// variable(0.017662706213291114)
}
