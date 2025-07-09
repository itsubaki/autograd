package model

import (
	"fmt"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Layer = (*L.LinearT)(nil)
	_ Layer = (*L.RNNT)(nil)
	_ Layer = (*L.LSTMT)(nil)
)

type Layer interface {
	First(x ...*variable.Variable) *variable.Variable
	Forward(x ...*variable.Variable) []*variable.Variable
	Params() L.Parameters
	Cleargrads()
}

type Model struct {
	Layers []Layer
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
