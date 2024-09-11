package model_test

import (
	"fmt"

	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/rand"
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
	m := model.NewLSTM(1, 1, model.WithLSTMSource(rand.Const()))

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
	// w variable([-0.007097596643213066])
	// w variable([[0.013515028138341746] [0.027030056276683492]])
	// w variable([[0.04252623292012907] [0.08505246584025813]])
	// b variable([0.05279536845172966])
	// b variable([0.013515028138341746])
	// b variable([0.04252623292012907])
	// w variable([[0.05279536845172966] [0.10559073690345933]])
	// b variable([-0.00757230286787535])
	// w variable([[-0.00757230286787535] [-0.0151446057357507]])
	// w variable([-0.0062508612982463485])
	// w variable([-0.017558407475346018])
	// w variable([0.0016808382857761302])
	// *layer.LinearT
	// b variable([2])
	// w variable([-1.1705639065492832])
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

func ExampleLSTM_Params() {
	m := model.NewLSTM(100, 1)

	x := variable.New(1, 2, 3)
	m.Forward(x)

	for k, p := range m.Params() {
		fmt.Println(k, variable.Shape(p))
	}

	// Unordered output:
	// 0.x2f.w [3 100]
	// 0.x2i.w [3 100]
	// 0.x2o.w [3 100]
	// 0.x2u.w [3 100]
	// 0.x2f.b [1 100]
	// 0.x2i.b [1 100]
	// 0.x2o.b [1 100]
	// 0.x2u.b [1 100]
	// 0.h2f.w [100 100]
	// 0.h2i.w [100 100]
	// 0.h2o.w [100 100]
	// 0.h2u.w [100 100]
	// 1.w [100 1]
	// 1.b [1 1]
}
