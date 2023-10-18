package model

import L "github.com/itsubaki/autograd/layer"

type Model struct {
	Layers []L.Layer
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
