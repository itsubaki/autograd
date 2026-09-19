package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	l := L.LSTM(2, L.WithLSTMSource(rand.Const()))

	x := variable.New(1.0).Reshape(1, 1)
	y := l.Forward(x)
	fmt.Println(y[0])

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v)
	}

	// Output:
	// variable[1 2]([-0.11594705 0.3410125])
	// h2f.w w[2 2]([0.40060148 -0.43303028 0.41711855 -0.260091])
	// h2i.w w[2 2]([0.7721305 -0.31383833 -0.17744653 0.6058784])
	// h2o.w w[2 2]([0.51316184 0.6022411 -1.4567161 -0.43950647])
	// h2u.w w[2 2]([-0.18634713 -0.2731477 -0.22971366 -0.97483426])
	// x2f.b b[1 2]([0 0])
	// x2f.w w[1 2]([1.4794952 -0.7850501])
	// x2i.b b[1 2]([0 0])
	// x2i.w w[1 2]([0.6683845 0.52241737])
	// x2o.b b[1 2]([0 0])
	// x2o.w w[1 2]([0.14228009 0.4955747])
	// x2u.b b[1 2]([0 0])
	// x2u.w w[1 2]([-0.3459245 2.3596406])
}

func ExampleLSTM_backward() {
	l := L.LSTM(2, L.WithLSTMSource(rand.Const()))

	x := variable.New(1.0).Reshape(1, 1)
	y := l.First(x)
	y.Backward()

	y = l.First(x)
	y.Backward()

	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// h2f.w variable[2 2]([0.0012131372 -0.004509243 -0.0035679643 0.013262159])
	// h2i.w variable[2 2]([0.0034706006 -0.007235676 -0.0102074025 0.021280887])
	// h2o.w variable[2 2]([0.01106945 -0.019636719 -0.03255642 0.05775366])
	// h2u.w variable[2 2]([-0.02077399 -0.0014846661 0.061098494 0.00436656])
	// x2f.b variable[1 2]([-0.010462855 0.038890537])
	// x2f.w variable[1 2]([-0.010462855 0.038890537])
	// x2i.b variable[1 2]([-0.08684848 0.18688211])
	// x2i.w variable[1 2]([-0.08684848 0.18688211])
	// x2o.b variable[1 2]([-0.14676197 0.30357784])
	// x2o.w variable[1 2]([-0.14676197 0.30357784])
	// x2u.b variable[1 2]([0.628041 0.024737671])
	// x2u.w variable[1 2]([0.628041 0.024737671])
}

func ExampleLSTM_cleargrads() {
	l := L.LSTM(3)

	x := variable.New(1.0).Reshape(1, 1)
	y := l.First(x)
	y.Backward()

	l.Cleargrads()
	for k, v := range l.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// h2f.w <nil>
	// h2i.w <nil>
	// h2o.w <nil>
	// h2u.w <nil>
	// x2f.b <nil>
	// x2f.w <nil>
	// x2i.b <nil>
	// x2i.w <nil>
	// x2o.b <nil>
	// x2o.w <nil>
	// x2u.b <nil>
	// x2u.w <nil>
}

func ExampleLSTMT_ResetState() {
	l := L.LSTM(3)

	x := variable.New(1.0).Reshape(1, 1)
	l.Forward(x)   // set hidden state
	l.ResetState() // reset hidden state
	l.Forward(x)   // h2h is not used

	for k := range l.Params().Seq2() {
		fmt.Println(k)
	}

	// Output:
	// h2f.w
	// h2i.w
	// h2o.w
	// h2u.w
	// x2f.b
	// x2f.w
	// x2i.b
	// x2i.w
	// x2o.b
	// x2o.w
	// x2u.b
	// x2u.w
}

func ExampleLSTM_batch() {
	l := L.LSTM(2, L.WithLSTMSource(rand.Const()))

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
	// h2f.w [2 2]
	// h2i.w [2 2]
	// h2o.w [2 2]
	// h2u.w [2 2]
	// x2f.b [1 2]
	// x2f.w [2 2]
	// x2i.b [1 2]
	// x2i.w [2 2]
	// x2o.b [1 2]
	// x2o.w [2 2]
	// x2u.b [1 2]
	// x2u.w [2 2]
}
