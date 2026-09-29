package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

func ExampleCrossEntropy() {
	x := variable.New(
		0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0,
	).Reshape(2, 10)

	t := variable.New(
		2,
		2,
	).Reshape(2, 1)

	y := F.CrossEntropy(x, t)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)

	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// Output:
	// variable(2.0694942)
	// [0.04916165 0.04676400 -0.41894615 0.04448330 0.04676400 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330]
	// [0.04916165 0.04676400 -0.45083836 0.04448330 0.04676400 0.04916165 0.04448330 0.08105386 0.04448330 0.04448330]
}

func ExampleOneHot() {
	v := F.OneHot([]int{0, 2, 2, 1}, 3, -100)

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
	x := tensor.New([]int{3, 5}, []float32{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})

	for _, v := range F.Logp(x, []int{0, 1, 3}, -100).Seq2() {
		fmt.Println(v)
	}

	// Output:
	// [1]
	// [7]
	// [14]
}

func ExampleCrossEntropy_double() {
	x := variable.New(
		0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0,
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0,
	).Reshape(2, 10)

	t := variable.New(
		2,
		2,
	).Reshape(2, 1)

	y := F.CrossEntropy(x, t)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)

	for _, row := range x.Grad.Data.Seq2() {
		fmt.Printf("%.8f\n", row)
	}

	// double
	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(F.Clip(0, 1)(x.Grad)) // NOTE: zeros..., why?

	// Output:
	// variable(2.0694942)
	// [0.04916165 0.04676400 -0.41894615 0.04448330 0.04676400 0.04916165 0.04448330 0.04916165 0.04448330 0.04448330]
	// [0.04916165 0.04676400 -0.45083836 0.04448330 0.04676400 0.04916165 0.04448330 0.08105386 0.04448330 0.04448330]
	// variable[2 10]([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])
}

func ExampleCrossEntropy_ignore() {
	x := variable.New(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	).Reshape(2, 3)

	t := variable.New(
		2,
		-100,
	).Reshape(2, 1)

	y := F.CrossEntropy(x, t)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(0.4076059)
	// variable[2 3]([0.09003057 0.24472848 -0.33475906 0 0 0])
}

func ExampleCrossEntropy_ignoreAll() {
	x := variable.New(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	).Reshape(2, 3)

	t := variable.New(
		-100,
		-100,
	).Reshape(2, 1)

	y := F.CrossEntropy(x, t)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(0)
	// variable[2 3]([0 0 0 0 0 0])
}
