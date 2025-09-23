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

	t := variable.New(2, 2)

	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward()
	fmt.Println(y)

	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// Output:
	// variable(2.069494302297095)
	// [0.04916165 0.04676401 -0.41894615 0.04448330 0.04676401 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330]
	// [0.04916165 0.04676401 -0.45083835 0.04448330 0.04676401 0.04916165 0.04448330 0.08105385 0.04448330 0.04448330]
}

func ExampleSoftmaxCrossEntropy_double() {
	x := variable.New(
		0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0,
	).Reshape(2, 10)

	t := variable.New(2, 2)

	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward(variable.Opts{CreateGraph: true})

	gx := x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(2.069494302297095)
	// variable[2 10]([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])
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

func ExampleSoftmaxCrossEntropy_nc() {
	x := variable.New(
		1.0, 2.0,
		0.5, 0.0,

		2.0, 1.0,
		0.0, 0.5,
	).Reshape(1, 2, 2, 2)

	t := variable.New(
		1, 0,
		0, 1,
	).Reshape(1, 2, 2)

	// Loss
	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward()
	fmt.Println(y)

	fmt.Println(x.Grad.Shape())
	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// Output:
	// variable(0.39366933584916486)
	// [1 2 2 2]
	// [0.06723536 -0.06723536]
	// [-0.09438517 0.09438517]
	// [-0.06723536 0.06723536]
	// [0.09438517 -0.09438517]
}

func ExampleSoftmaxCrossEntropy_nchw() {
	x := variable.New(
		// batch 0
		2.0, 1.0,
		0.1, 1.0,

		2.0, 0.2,
		0.1, 0.2,

		3.0, 0.5,
		0.3, 0.1,

		// batch 1
		0.5, 1.5,
		0.2, 0.1,

		2.0, 0.5,
		0.3, 0.2,

		0.1, 0.1,
		0.4, 2.0,
	).Reshape(2, 3, 2, 2)

	t := variable.New(
		0, 1,
		2, 0,

		1, 0,
		2, 2,
	).Reshape(2, 2, 2)

	// Loss
	y := F.SoftmaxCrossEntropy(x, t)
	y.Backward()
	fmt.Println(y)

	fmt.Println(x.Grad.Shape())
	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// Output:
	// variable(0.76009170620603)
	// [2 3 2 2]
	// [-0.04262486 0.03030412]
	// [0.01232074 0.02999320]
	// [-0.04347002 0.01347682]
	// [0.00616392 0.00681218]
	// [-0.01297610 -0.07478005]
	// [0.04111662 0.03366344]
	// [0.02803256 -0.04879960]
	// [0.02076703 -0.11138006]
	// [0.09106149 0.02031857]
	// [0.04589568 0.04152812]
	// [-0.08742380 0.01383393]
	// [0.01867386 -0.03250779]
}
