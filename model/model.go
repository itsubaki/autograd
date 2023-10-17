package model

import (
	"github.com/itsubaki/autograd/layer"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Layer = (L.Linear(1))
	_ Layer = (L.RNN(1))
)

type Layer interface {
	First(x ...*variable.Variable) *variable.Variable
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() []layer.Parameter
	Cleargrads()
}

type Model struct {
	Layers []Layer
}

func (m Model) Params() []L.Parameter {
	params := make([]L.Parameter, 0)
	for _, l := range m.Layers {
		params = append(params, l.Params()...)
	}

	return params
}

func (m *Model) Cleargrads() {
	for _, l := range m.Layers {
		l.Cleargrads()
	}
}
