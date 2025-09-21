package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleRNN() {
	l := L.RNN(2, L.WithRNNSource(rand.Const()))

	x := variable.New(1.0)
	y := l.Forward(x)
	fmt.Println(y[0])

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v)
	}

	// Output:
	// variable[1 2]([0.7975914906443963 -0.41681777230979594])
	// h2h.w w[2 2]([0.4006014980172961 -0.4330302800303532 0.4171185512277987 -0.260091010167568])
	// x2h.b b[1 2]([0 0])
	// x2h.w w[1 2]([1.0919575041640825 -0.4438344619606553])
}

func ExampleRNN_backward() {
	l := L.RNN(2, L.WithRNNSource(rand.Const()))

	x := variable.New(1.0)
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
	// x2h.b variable[1 2]([0.3638478140516499 0.8262629446866991])
	// x2h.w variable[1 2]([0.3638478140516499 0.8262629446866991])
	// .
	// h2h.w variable[2 2]([0.22839718521198515 0.5180241623919872 -0.11935935508160048 -0.2707171276318745])
	// x2h.b variable[1 2]([0.5896143925532906 1.4348651282031906])
	// x2h.w variable[1 2]([0.5896143925532906 1.4348651282031906])
}

func ExampleRNN_cleargrads() {
	l := L.RNN(3)

	x := variable.New(1)
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

	x := variable.New(1)
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
