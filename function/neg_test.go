package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleNeg() {
	// p139
	x := variable.New(3.0)
	y := F.Neg(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[-3]
	// variable[-1]
}

func ExampleNeg_higher() {
	x := variable.New(2.0)
	y := F.Neg(x)
	y.Backward(variable.Opts{Retain: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 1; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{Retain: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable[-2]
	// variable[-1]
	// <nil>
}
