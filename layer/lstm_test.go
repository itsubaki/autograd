package layer_test

import (
	"fmt"
	"math/rand"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	l := L.LSTM(2, L.LSTMOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.Forward(x)
	fmt.Printf("%.4f\n", y[0].Data)

	for k, v := range l.Params() {
		fmt.Println(k, v.Data)
	}

	// Unordered output:
	// [[0.2219 -0.1172]]
	// x2f.w [[0.7001209024399848 0.4315307186960532]]
	// x2f.b [[0 0]]
	// x2i.w [[0.9996261210112625 -1.5239676725278932]]
	// x2i.b [[0 0]]
	// x2o.w [[-0.31653724289408824 1.8894642062634817]]
	// x2o.b [[0 0]]
	// x2u.w [[1.1007291937500208 -0.9927431907514367]]
	// x2u.b [[0 0]]
	// h2f.w [[-0.872398773723865 -0.08934118160368779] [-0.36839879422376975 1.6162474880131052]]
	// h2i.w [[0.2282577831242846 0.4172405804180962] [0.11229402998366894 0.6994715017692726]]
	// h2o.w [[-0.5170951797056471 0.4853445075750822] [1.1210498926486498 0.5927010790465582]]
	// h2u.w [[0.9184191709616603 0.3728986958482689] [0.5179146525617053 -0.7588527289244226]]
}

func ExampleLSTM_backward() {
	l := L.LSTM(2, L.LSTMOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1)
	y := l.First(x)
	y.Backward()

	y = l.First(x)
	y.Backward()

	for k, v := range l.Params() {
		fmt.Println(k, v.Grad)
	}

	// Unordered output:
	// x2f.w variable([0.020823321307053687 -0.028140599335702825])
	// x2f.b variable([0.020823321307053687 -0.028140599335702825])
	// x2i.w variable([0.08422779970496914 -0.23369417004444376])
	// x2i.b variable([0.08422779970496914 -0.23369417004444376])
	// x2o.w variable([0.293051903606226 -0.03742522138109533])
	// x2o.b variable([0.293051903606226 -0.03742522138109533])
	// x2u.w variable([0.1305869202756255 0.18447813513522493])
	// x2u.b variable([0.1305869202756255 0.18447813513522493])
	// h2f.w variable([[0.004621665085097407 -0.006245709966520938] [-0.0024394546129570265 0.0032966746201818433]])
	// h2i.w variable([[0.005566014529823725 -0.018626324963080025] [-0.0029379108114836816 0.009831537663803807]])
	// h2o.w variable([[0.03899972195488069 -0.0047924374902220705] [-0.020585232784099496 0.002529593453348216]])
	// h2u.w variable([[0.007118188488063291 0.018281608280867702] [-0.0037571951717348686 -0.009649585773066927]])
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
