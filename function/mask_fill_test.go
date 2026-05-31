package function_test

import (
	"fmt"
	"math"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMaskFill() {
	mask := tensor.Tril(tensor.Ones[float64](3, 3))
	f := F.MaskFill(mask, math.Inf(-1))

	x := variable.New(
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	).Reshape(3, 3)

	y := f(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 3]([1 -Inf -Inf 4 5 -Inf 7 8 9])
	// variable[3 3]([1 0 0 1 1 0 1 1 1])
}
