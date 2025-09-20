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

	shape := x.Grad.Shape()
	for i := range shape[0] {
		row := make([]float64, shape[1])
		for j := range shape[1] {
			row[j] = x.Grad.At(i, j)
		}

		fmt.Printf("%.8f\n", row)
	}

	// Output:
	// variable(2.069494302297095)
	// [0.04916165 0.04676401 -0.41894615 0.04448330 0.04676401 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330]
	// [0.04916165 0.04676401 -0.45083835 0.04448330 0.04676401 0.04916165 0.04448330 0.08105385 0.04448330 0.04448330]
}

func ExampleOneHot() {
	v := F.OneHot([]float64{0, 2, 2, 1}, 3)

	shape := v.Shape
	for i := range shape[0] {
		row := make([]float64, shape[1])
		for j := range shape[1] {
			row[j] = v.At(i, j)
		}

		fmt.Println(row)
	}

	// Output:
	// [1 0 0]
	// [0 0 1]
	// [0 0 1]
	// [0 1 0]
}

func ExampleLogp() {
	A := tensor.New([]int{3, 5}, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	for _, v := range F.Logp(A, []int{0, 1, 3}).Seq2() {
		fmt.Println(v)
	}

	// Output:
	// [1]
	// [7]
	// [14]
}
