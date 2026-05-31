package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleGELU() {
	x := variable.New(
		-10, 0, 10,
	).Reshape(1, 3)

	y := F.GELU(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([-0 0 10])
	// variable[1 3]([0 0.5 1])
}
