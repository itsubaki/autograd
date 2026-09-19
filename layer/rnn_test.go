package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleRNN() {
	l := L.RNN(2, L.WithRNNSource(rand.Const()))

	x := variable.New(1.0).Reshape(1, 1)
	y := l.Forward(x)
	fmt.Println(y[0])

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v)
	}

	// Output:
	// variable[1 2]([0.79759145 -0.41681775])
	// h2h.w w[2 2]([0.40060148 -0.43303028 0.41711855 -0.260091])
	// x2h.b b[1 2]([0 0])
	// x2h.w w[1 2]([1.0919574 -0.44383445])
}

func ExampleRNN_backward() {
	l := L.RNN(2, L.WithRNNSource(rand.Const()))

	x := variable.New(1.0).Reshape(1, 1)
	y := l.First(x)
	y.Backward()

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}
	fmt.Println(".")

	y = l.First(x)
	y.Backward()

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// h2h.w <nil>
	// x2h.b variable[1 2]([0.3638479 0.82626295])
	// x2h.w variable[1 2]([0.3638479 0.82626295])
	// .
	// h2h.w variable[2 2]([0.22839727 0.51802415 -0.1193594 -0.2707171])
	// x2h.b variable[1 2]([0.5896146 1.4348652])
	// x2h.w variable[1 2]([0.5896146 1.4348652])
}

func ExampleRNN_cleargrads() {
	l := L.RNN(3)

	x := variable.New(1).Reshape(1, 1)
	y := l.First(x)
	y.Backward()

	l.Cleargrads()
	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// h2h.w <nil>
	// x2h.b <nil>
	// x2h.w <nil>
}

func ExampleRNNT_ResetState() {
	l := L.RNN(3)

	x := variable.New(1).Reshape(1, 1)
	l.Forward(x)   // set hidden state
	l.ResetState() // reset hidden state
	l.Forward(x)   // h2h is not used

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Name)
	}

	// Output:
	// h2h.w w
	// x2h.b b
	// x2h.w w
}

func ExampleRNN_batch() {
	l := L.RNN(2, L.WithRNNSource(rand.Const()))

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
	// [2 2 2]
	// [2 2 2]
	// h2h.w [2 2]
	// x2h.b [1 2]
	// x2h.w [2 2]
}
