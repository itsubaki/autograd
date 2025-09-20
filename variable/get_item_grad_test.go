package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleGetItemGrad() {
	gy := variable.NewOf(
		[]float64{1, 1, 1},
		[]float64{1, 1, 1},
		[]float64{1, 1, 1},
	)

	y := variable.GetItemGrad([]int{0, 0, 1}, []int{2, 3})(gy)
	y.Backward()

	fmt.Println(y)
	fmt.Println(gy.Grad)

	// Output:
	// variable[2 3]([2 2 2 1 1 1])
	// variable[3 3]([1 1 1 1 1 1 1 1 1])
}
