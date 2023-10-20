package layer_test

import (
	"fmt"
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	l := L.LSTM(5, L.LSTMOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data)

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Data)
	}

	// Unordered output:
	// [[-0.3892 0.2199 0.2263 0.1525 -0.3650]]
	// x2f.w [[-1.233758177597947 -0.12634751070237293 -0.5209945711531503 2.28571911769958 0.3228052526115799]]
	// x2f.b [[0 0 0 0 0]]
	// x2i.w [[0.5900672875996937 0.15880774017643562 0.9892020842955818 -0.731283016177479 0.6863807850359727]]
	// x2i.b [[0 0 0 0 0]]
	// x2o.w [[1.585403962280623 0.8382059044208106 1.2988408475174342 0.5273583930598617 0.7324419258045132]]
	// x2o.b [[0 0 0 0 0]]
	// x2u.w [[-1.0731798210887524 0.7001209024399848 0.4315307186960532 0.9996261210112625 -1.5239676725278932]]
	// x2u.b [[0 0 0 0 0]]
}

func ExampleLSTM_backward() {
	l := L.LSTM(3, L.LSTMOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	y = l.First(x)
	y.Backward()

	for k, v := range l.FlattenParams() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2f.w variable([0.039205406735673486 0.05322207827710336 0.017389639147083143])
	// x2f.b variable([0.039205406735673486 0.05322207827710336 0.017389639147083143])
	// x2i.w variable([0.028880122544234135 0.2089185279948017 0.07392173355499831])
	// x2i.b variable([0.028880122544234135 0.2089185279948017 0.07392173355499831])
	// x2o.w variable([0.19016768679739787 0.18632840300075368 0.16626324411958548])
	// x2o.b variable([0.19016768679739787 0.18632840300075368 0.16626324411958548])
	// x2u.w variable([0.6334057457218616 0.06845208692222428 0.10956533339103915])
	// x2u.b variable([0.6334057457218616 0.06845208692222428 0.10956533339103915])
	// h2f.w variable([[0.010440296708609366 0.01417289947807621 0.00463081517238793] [0.013942834753706454 0.01892766086243065 0.0061843731577396685] [0.005275738084914379 0.007161913845656344 0.002340064526041666]])
	// h2o.w variable([[0.020651816032868246 0.03459710308687688 0.027776726556092996] [0.027580141287821828 0.04620382971486683 0.03709533542769211] [0.010435869344345565 0.01748276504753257 0.014036261445078385]])
	// h2i.w variable([[0.0014461577335875108 0.024462384957841195 0.009181458895284218] [0.0019313185122964826 0.03266908984181207 0.01226167873843382] [0.000730778985006197 0.012361443316404524 0.004639616451607423]])
	// h2u.w variable([[0.09540186036770151 0.005746568624155163 0.006341633540442708] [0.12740752599551514 0.00767444249561965 0.00846914134634135] [0.04820890077755342 0.002903882120804941 0.0032045830232290784]])
}

func ExampleLSTM_cleargrads() {
	l := L.LSTM(3)

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	l.Cleargrads()
	for k, v := range l.FlattenParams() {
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
}

func ExampleLSTMT_ResetState() {
	l := L.LSTM(3)

	x := variable.New(1)
	l.Forward(x)   // set hidden state
	l.ResetState() // reset hidden state
	l.Forward(x)   // h2h is not used

	for k := range l.FlattenParams() {
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
}
