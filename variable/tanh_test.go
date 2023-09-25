package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func Example_tanh() {
	// p249
	x := variable.New(1.0)
	y := variable.Tanh(x)
	y.Backward()

	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	y.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// 1-tanh(1)^2                                =  0.41997434161
	// −2*tanh(1)*(1−tanh(1)^2)                   = -0.63970000844
	// 4*tanh(1)^2*sech(1)^2-2*sech(1)^4          =  0.62162668077
	// 16*tanh(1)*sech(1)^4-8*tanh(1)^3*sech(1)^2 =  0.66509104475

	// Output:
	// variable[0.41997434161402614]
	// variable[-0.6397000084492246]
}
