package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func Example_tanh() {
	// p249
	x := variable.New(1.0)
	y := F.Tanh(x)
	y.Backward()

	fmt.Println(x.Grad)

	for i := 0; i < 2; i++ {
		gx := x.Grad
		x.Cleargrad()
		y.Cleargrad()
		gx.Backward()
		fmt.Println(x.Grad)
	}

	// 1-tanh(1)^2                          =  0.41997434161
	// −2*tanh(1)*(1−tanh(1)^2)             = -0.63970000844
	// -2*(1-3*(tanh(1.0))^2)/(cosh(1.0))^2 =  0.62162668077
	// -8*((cosh(1)^2-3)tanh(1)/cosh(1)^4   =  0.66509104475

	// Output:
	// variable[0.41997434161402614]
	// variable[-0.6397000084492246]
	// variable[0.6216266807712962]
}
