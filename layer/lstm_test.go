package layer_test

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	l := L.LSTM(2, L.WithLSTMSource(rand.Const()))

	x := variable.New(1)
	y := l.Forward(x)
	fmt.Printf("%v\n", y[0].Data)

	for k, v := range l.Params() {
		fmt.Println(k, v.Data)
	}

	// Unordered output:
	// [[-0.1159470564250778 0.34101249498339026]]
	// h2i.w [[0.7721305559619605 -0.3138383577766621] [-0.1774465290440516 0.6058784343349307]]
	// h2u.w [[-0.1863471272377204 -0.2731476909916124] [-0.22971366432393717 -0.9748342951297205]]
	// x2f.b [[0 0]]
	// x2i.b [[0 0]]
	// x2i.w [[0.6683845105491564 0.5224173744445297]]
	// x2u.w [[-0.3459245000660268 2.3596405779473866]]
	// h2f.w [[0.4006014980172961 -0.4330302800303532] [0.4171185512277987 -0.260091010167568]]
	// x2f.w [[1.4794951368388003 -0.7850500761442023]]
	// x2o.b [[0 0]]
	// x2o.w [[0.14228008038102535 0.49557472469859015]]
	// x2u.b [[0 0]]
	// h2o.w [[0.5131618329299671 0.6022410956921772] [-1.4567160689766936 -0.4395064776168111]]

}

func ExampleLSTM_backward() {
	l := L.LSTM(2, L.WithLSTMSource(rand.Const()))

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	y = l.First(x)
	y.Backward()

	for k, v := range l.Params() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2o.b variable[1 2]([-0.1467619739042175 0.3035778427291682])
	// x2o.w variable[1 2]([-0.1467619739042175 0.3035778427291682])
	// x2u.b variable[1 2]([0.6280410802257455 0.024737661571790713])
	// h2i.w variable[2 2]([[0.0034706005215721916 -0.007235676351057161] [-0.010207401373028816 0.02128088561662388]])
	// x2f.b variable[1 2]([-0.010462855229162323 0.038890543079566985])
	// x2i.b variable[1 2]([-0.086848478802109 0.18688211132241805])
	// x2i.w variable[1 2]([-0.086848478802109 0.18688211132241805])
	// x2u.w variable[1 2]([0.6280410802257455 0.024737661571790713])
	// h2f.w variable[2 2]([[0.0012131372656231042 -0.004509243992848472] [-0.003567964366346655 0.013262161126822158]])
	// h2o.w variable[2 2]([[0.01106944986063837 -0.019636718150272264] [-0.03255641696698899 0.05775365460904995]])
	// h2u.w variable[2 2]([[-0.020773992595966805 -0.001484667253885979] [0.06109849843833466 0.0043665626371028115]])
	// x2f.w variable[1 2]([-0.010462855229162323 0.038890543079566985])

}

func ExampleLSTM_cleargrads() {
	l := L.LSTM(3)

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	l.Cleargrads()
	for k, v := range l.Params() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2f.w <nil>
	// x2f.b <nil>
	// x2i.w <nil>
	// x2i.b <nil>
	// x2o.w <nil>
	// x2o.b <nil>
	// x2u.w <nil>
	// x2u.b <nil>
	// h2f.w <nil>
	// h2i.w <nil>
	// h2o.w <nil>
	// h2u.w <nil>
}

func ExampleLSTMT_ResetState() {
	l := L.LSTM(3)

	x := variable.New(1)
	l.Forward(x)   // set hidden state
	l.ResetState() // reset hidden state
	l.Forward(x)   // h2h is not used

	for k := range l.Params() {
		fmt.Println(k)
	}

	// Unordered output:
	// x2f.w
	// x2f.b
	// x2i.w
	// x2i.b
	// x2o.w
	// x2o.b
	// x2u.w
	// x2u.b
	// h2f.w
	// h2i.w
	// h2o.w
	// h2u.w
}
