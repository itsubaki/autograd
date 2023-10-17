package model

import (
	"github.com/itsubaki/autograd/dot"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

type Model struct {
	L.Layer
}

func (m Model) graph(y *variable.Variable, opts ...dot.Opts) []string {
	out := make([]string, 0)
	for _, txt := range dot.Graph(y, opts...) {
		out = append(out, txt)
	}

	return out
}
