package model_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/rand"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMLP() {
	m := model.NewMLP([]int{1, 2, 3})

	for _, l := range m.Layers {
		fmt.Printf("%T\n", l)
	}

	// Output:
	// *layer.LinearT
	// *layer.LinearT
	// *layer.LinearT
}

func ExampleMLP_backward() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()

	for _, p := range m.Params() {
		fmt.Println(p.Name, p.Grad)
	}

	// Unordered output:
	// w variable([[0 0 -0.11785627150007956 -0.17275376822987032 -0.1452836777854009] [0 0 -0.23571254300015912 -0.34550753645974064 -0.2905673555708018]])
	// b variable([1])
	// w variable([[0] [0] [1.62887541989766] [0.7662326556923662] [1.9766127473463149]])
	// b variable([0 0 -0.11785627150007956 -0.17275376822987032 -0.1452836777854009])
}

func ExampleMLP_cleargrads() {
	m := model.NewMLP([]int{5, 1},
		model.WithMLPSource(rand.Const()),
		model.WithMLPActivation(F.ReLU),
	)

	x := variable.New(1, 2)
	y := m.Forward(x)
	y.Backward()
	m.Cleargrads()

	for _, p := range m.Params() {
		fmt.Println(p.Name, p.Grad)
	}

	// Unordered output:
	// b <nil>
	// w <nil>
	// b <nil>
	// w <nil>
}

func ExampleMLP_Params() {
	m := model.NewMLP([]int{5, 1})

	x := variable.New(1, 2)
	m.Forward(x) // gen w

	for k, p := range m.Params() {
		fmt.Println(k, p.Shape())
	}

	// Unordered output:
	// 0.w [2 5]
	// 0.b [1 5]
	// 1.w [5 1]
	// 1.b [1 1]
}
