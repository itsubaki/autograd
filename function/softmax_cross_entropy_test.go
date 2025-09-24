package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmaxCrossEntropy() {
	x := variable.New(
		0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0,
	).Reshape(2, 10)

	t := variable.New(
		2,
		2,
	).Reshape(2, 1)

	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)

	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// double
	gx := x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(F.Clip(0, 1)(x.Grad))

	// Output:
	// variable(2.069494302297095)
	// [0.04916165 0.04676401 -0.41894615 0.04448330 0.04676401 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330]
	// [0.04916165 0.04676401 -0.45083835 0.04448330 0.04676401 0.04916165 0.04448330 0.08105385 0.04448330 0.04448330]
	// variable[2 10]([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])
}

func ExampleOneHot() {
	v := F.OneHot([]int{0, 2, 2, 1}, 3)

	for _, row := range v.Seq2() {
		fmt.Println(row)
	}

	// Output:
	// [1 0 0]
	// [0 0 1]
	// [0 0 1]
	// [0 1 0]
}

func ExampleLogp() {
	x := tensor.New([]int{3, 5}, []float64{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	for _, v := range F.Logp(x, []int{0, 1, 3}).Seq2() {
		fmt.Println(v)
	}

	// Output:
	// [1]
	// [7]
	// [14]
}
