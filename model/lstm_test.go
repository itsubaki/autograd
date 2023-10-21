package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/variable"
)

func ExampleLSTM() {
	m := model.NewLSTM(2, 3)

	for _, l := range m.Layers {
		fmt.Printf("%T\n", l)
	}

	// Output:
	// *layer.LSTMT
	// *layer.LinearT
}

func ExampleLSTM_backward() {
	m := model.NewLSTM(1, 1, model.LSTMOpts{
		Source: rand.NewSource(1),
	})

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()
	y = m.Forward(x)
	y.Backward()

	for _, l := range m.Layers {
		fmt.Printf("%T\n", l)
		for _, p := range l.Params() {
			fmt.Println(p.Name, p.Grad)
		}
	}

	// Unordered output:
	// *layer.LSTMT
	// w variable([0.011226806999443534])
	// w variable([0.11294101620706003])
	// w variable([0.00036758770803267443])
	// w variable([0.013679446277302425])
	// b variable([0.03359993499751152])
	// b variable([0.016635625479843978])
	// b variable([0.09921510586976715])
	// b variable([0.4532075079866421])
	// w variable([[0.03359993499751152] [0.06719986999502305]])
	// w variable([[0.09921510586976715] [0.1984302117395343]])
	// w variable([[0.4532075079866421] [0.9064150159732842]])
	// w variable([[0.016635625479843978] [0.033271250959687956]])
	// *layer.LinearT
	// b variable([2])
	// w variable([0.8926291447755661])
}

func ExampleLSTM_ResetState() {
	m := model.NewLSTM(1, 1)

	x := variable.New(1, 2)
	m.Forward(x)
	m.ResetState()
	m.Forward(x)

	for _, p := range m.Params() {
		fmt.Println(p.Name, p.Grad)
	}

	// Unordered output:
	// w <nil>
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
	// w <nil>
	// w <nil>
	// w <nil>
}
