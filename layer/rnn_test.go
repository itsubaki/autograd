package layer_test

import (
	"fmt"
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleRNN() {
	l := L.RNN(2, L.RNNOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data)

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Data)
	}

	// Unordered output:
	// [[0.3120 0.5299]]
	// x2h.w [[0.3228052526115799 0.5900672875996937]]
	// x2h.b [[0 0]]
	// h2h.w [[-0.872398773723865 -0.08934118160368779] [-0.36839879422376975 1.6162474880131052]]
}

func ExampleRNN_backward() {
	l := L.RNN(2, L.RNNOpts{
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
	// x2h.w variable([0.902630265225777 0.719159357118377])
	// x2h.b variable([0.902630265225777 0.719159357118377])
	// h2h.w <nil>
	// .
	// x2h.w variable([1.0939291043139359 0.9118192171630227])
	// x2h.b variable([1.0939291043139359 0.9118192171630227])
	// h2h.w variable([[0.3056022228722387 0.0652456710508777] [0.5190085287937879 0.11080763557284856]])
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
	// h2h.w <nil>
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
	// w
}
