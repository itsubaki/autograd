package model

import (
	"github.com/itsubaki/autograd/dot"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type Model struct {
	L.Layer
}

func (m Model) graph(y *variable.Variable, opt ...dot.Opt) []string {
	out := make([]string, 0)
	for _, txt := range dot.Graph(y, opt...) {
		out = append(out, txt)
	}

	return out
}
