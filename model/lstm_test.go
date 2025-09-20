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
		for k, v := range l.Params().Seq2() {
			fmt.Println(k, v.Grad)
		}
	}

	// Output:
	// *layer.LSTMT
	// h2f.w variable(-0.007097596643213066)
	// h2i.w variable(-0.0062508612982463485)
	// h2o.w variable(-0.017558407475346018)
	// h2u.w variable(0.0016808382857761302)
	// x2f.b variable(0.013515028138341746)
	// x2f.w variable[2 1]([[0.013515028138341746] [0.027030056276683492]])
	// x2i.b variable(0.04252623292012907)
	// x2i.w variable[2 1]([[0.04252623292012907] [0.08505246584025813]])
	// x2o.b variable(0.05279536845172966)
	// x2o.w variable[2 1]([[0.05279536845172966] [0.10559073690345933]])
	// x2u.b variable(-0.00757230286787535)
	// x2u.w variable[2 1]([[-0.00757230286787535] [-0.0151446057357507]])
	// *layer.LinearT
	// b variable(2)
	// w variable(-1.1705639065492832)
}

func ExampleLSTM_ResetState() {
	m := model.NewLSTM(1, 1)

	x := variable.New(1, 2)
	m.Forward(x)
	m.ResetState()
	m.Forward(x)

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Grad)
	}

	// Output:
	// 0.h2f.w <nil>
	// 0.h2i.w <nil>
	// 0.h2o.w <nil>
	// 0.h2u.w <nil>
	// 0.x2f.b <nil>
	// 0.x2f.w <nil>
	// 0.x2i.b <nil>
	// 0.x2i.w <nil>
	// 0.x2o.b <nil>
	// 0.x2o.w <nil>
	// 0.x2u.b <nil>
	// 0.x2u.w <nil>
	// 1.b <nil>
	// 1.w <nil>
}

func ExampleLSTM_Params() {
	m := model.NewLSTM(100, 1)

	x := variable.New(1, 2, 3)
	m.Forward(x)

	for k, v := range m.Params().Seq2() {
		fmt.Println(k, v.Shape())
	}

	// Output:
	// 0.h2f.w [100 100]
	// 0.h2i.w [100 100]
	// 0.h2o.w [100 100]
	// 0.h2u.w [100 100]
	// 0.x2f.b [1 100]
	// 0.x2f.w [3 100]
	// 0.x2i.b [1 100]
	// 0.x2i.w [3 100]
	// 0.x2o.b [1 100]
	// 0.x2o.w [3 100]
	// 0.x2u.b [1 100]
	// 0.x2u.w [3 100]
	// 1.b [1 1]
	// 1.w [100 1]
}
