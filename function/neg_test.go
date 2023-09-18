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
