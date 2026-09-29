package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLinear() {
	l := L.Linear(5, L.WithSource(rand.Const()))

	x := variable.New(
		1, 2, 3,
	).Reshape(1, 3)

	y := l.Forward(x)
	fmt.Printf("%v, %.4f\n", y[0].Shape(), y[0].Data.Data)

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v)
	}

	// Output:
	// [1 5], [-3.7536 -1.7199 0.8735 -0.0434 1.0512]
	// b b[1 5]([0 0 0 0 0])
	// w w[3 5]([0.32708976 -0.35356775 0.34057587 -0.2123634 0.6304419 -0.25624794 -0.14488448 0.49469766 0.41899487 0.49172777 -1.1894038 -0.35885555 -0.1521518 -0.22302416 -0.18756042])
}

func ExampleLinear_inSize() {
	l := L.Linear(5,
		L.WithSource(rand.Const()),
		L.WithInSize(3),
	)

	x := variable.New(
		1, 2, 3,
	).Reshape(1, 3)

	y := l.Forward(x)
	fmt.Printf("%v, %.4f\n", y[0].Shape(), y[0].Data.Data)

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v)
	}

	// Output:
	// [1 5], [-3.7536 -1.7199 0.8735 -0.0434 1.0512]
	// b b[1 5]([0 0 0 0 0])
	// w w[3 5]([0.32708976 -0.35356775 0.34057587 -0.2123634 0.6304419 -0.25624794 -0.14488448 0.49469766 0.41899487 0.49172777 -1.1894038 -0.35885555 -0.1521518 -0.22302416 -0.18756042])
}

func ExampleLinear_nobias() {
	l := L.Linear(5, L.WithNoBias())

	x := variable.New(
		1, 2, 3,
	).Reshape(1, 3)

	l.Forward(x)
	for _, v := range l.Params().Seq2() {
		fmt.Println(v.Name)
	}

	// Output:
	// w
}

func ExampleLinear_backward() {
	l := L.Linear(5)

	y := l.Forward(variable.New(
		1, 2, 3,
	).Reshape(1, 3))

	y[0].Backward()
	for _, v := range l.Params().Seq2() {
		fmt.Println(v.Name, v.Grad)
	}

	y = l.Forward(variable.New(
		1, 2, 3,
	).Reshape(1, 3))

	y[0].Backward()
	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// b variable[1 5]([1 1 1 1 1])
	// w variable[3 5]([1 1 1 1 1 2 2 2 2 2 3 3 3 3 3])
	// b variable[1 5]([2 2 2 2 2])
	// w variable[3 5]([2 2 2 2 2 4 4 4 4 4 6 6 6 6 6])
}

func ExampleLinear_batch() {
	l := L.Linear(5, L.WithSource(rand.Const()))

	x := variable.New(
		1, 2,
		3, 4,

		5, 6,
		7, 8,
	).Reshape(2, 2, 2)

	y := l.Forward(x)
	y[0].Backward()

	fmt.Println(y[0].Shape())
	fmt.Println(x.Grad.Shape())

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// [2 2 5]
	// [2 2 2]
	// b [1 5]
	// w [2 5]
}
