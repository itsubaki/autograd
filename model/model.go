package model

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
)

type Model struct {
	Layers []L.Layer
}

func (m Model) Params() L.Parameters {
	params := make(L.Parameters, 0)
	for i, l := range m.Layers {
		for k, p := range l.Params() {
			params[fmt.Sprintf("%d.%s", i, k)] = p
		}
	}

	return params
}

func (m *Model) Cleargrads() {
	for _, l := range m.Layers {
		l.Cleargrads()
	}
}
