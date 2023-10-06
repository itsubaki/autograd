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
	// variable([0.01798620996209156])
	// variable([0.11920292202211755])
	// variable([0.5])
	// variable([0.8807970779778823])
	// variable([0.9820137900379085])
}
