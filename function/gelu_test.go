package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleGELU() {
	x := variable.New(
		-10, -3, 0, 3, 10,
	).Reshape(1, 5)

	y := F.GELU(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 5]([-0 -0.0036373920817729943 0 2.996362607918227 10])
	// variable[1 5]([0 -0.011584166630969648 0.5 1.0115841666309695 1])
}
