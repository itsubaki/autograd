package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleGELU() {
	x := variable.New(
		-3, -2, -1, 0,
		1, 2, 3, 4,
	).Reshape(2, 4)

	y := F.GELU(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 4]([-0.0036373920817729943 -0.04540230591222494 -0.15880800939172324 0 0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075])
	// variable[2 4]([-0.011584166630969516 -0.08609925662361825 -0.08296408384578252 0.5 1.0829640838457826 1.0860992566236183 1.0115841666309695 1.000335123197122])
}
