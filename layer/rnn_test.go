package layer_test

import (
	"fmt"
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleRNN() {
	l := L.RNN(5, L.RNNOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data)

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Data)
	}

	// Unordered output:
	// [[-0.8437 -0.1257 -0.4785 0.9795 0.3120]]
	// x2h.w [[-1.233758177597947 -0.12634751070237293 -0.5209945711531503 2.28571911769958 0.3228052526115799]]
	// x2h.b [[0 0 0 0 0]]
}

func ExampleRNN_backward() {
	l := L.RNN(3, L.RNNOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Grad)
	}
	fmt.Println(".")

	y = l.First(x)
	y.Backward()

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2h.w variable([0.28822771944106673 0.984204675359483 0.7710690810213434])
	// x2h.b variable([0.28822771944106673 0.984204675359483 0.7710690810213434])
	// .
	// x2h.w variable([0.39396398439363184 1.6894261397851316 1.764807089541906])
	// x2h.b variable([0.39396398439363184 1.6894261397851316 1.764807089541906])
	// h2h.w variable([[-0.020396585815502664 -0.4758483540449156 -0.3614345140035484] [-0.003038443883017692 -0.07088630095596118 -0.053842270374037576] [-0.01156749092376784 -0.269867298688821 -0.204979916643439]])
}

func ExampleRNN_cleargrads() {
	l := L.RNN(3)

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	l.Cleargrads()
	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2h.w <nil>
	// x2h.b <nil>
}

func ExampleRNNT_ResetState() {
	l := L.RNN(3)

	x := variable.New(1)
	l.Forward(x)   // set hidden state
	l.ResetState() // reset hidden state
	l.Forward(x)   // h2h is not used

	for _, v := range l.Params() {
		fmt.Println(v.Name)
	}

	// Unordered output:
	// w
	// b
}
