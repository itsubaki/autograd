package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExamplePow() {
	v := variable.New(2.0)
	y := F.Pow(3.0)(v)
	y[0].Backward()

	fmt.Println(y)
	fmt.Println(v.Grad)

	// Output:
	// [variable([8])]
	// [12]
}
