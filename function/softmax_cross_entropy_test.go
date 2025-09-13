package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmaxCrossEntropy() {
	x := variable.NewOf(
		[]float64{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0},
		[]float64{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0},
	)
	t := variable.New(2, 2)

	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward()

	fmt.Println(y)
	for i := range 2 {
		for j := range 10 {
			fmt.Printf("%f ", x.Grad.Data.At(i, j))
		}
		fmt.Println()
	}

	// Output:
	// variable([2.069494302297095])
	// 0.04916165 0.04676401 -0.41894615 0.04448330 0.04676401 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330
	// 0.04916165 0.04676401 -0.45083835 0.04448330 0.04676401 0.04916165 0.04448330 0.08105385 0.04448330 0.04448330
}

func ExampleOneHot() {
	v := F.OneHot([]float64{0, 2, 2, 1}, 3)

	fmt.Println(v.At(0, 0), v.At(0, 1), v.At(0, 2))
	fmt.Println(v.At(1, 0), v.At(1, 1), v.At(1, 2))
	fmt.Println(v.At(2, 0), v.At(2, 1), v.At(2, 2))
	fmt.Println(v.At(3, 0), v.At(3, 1), v.At(3, 2))

	// Output:
	// 1 0 0
	// 0 0 1
	// 0 0 1
	// 0 1 0
}

func ExampleLogp() {
	A := tensor.New([]int{3, 5}, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	v := F.Logp(A, []int{0, 1, 3})

	fmt.Println(v.Shape)
	fmt.Println(v.Data)

	// Output:
	// [3 1]
	// [1 7 14]
}
