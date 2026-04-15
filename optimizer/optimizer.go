package optimizer

import (
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/model"
)

var (
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.LSTM)(nil)
)

var (
	_ Hook = hook.WeightDecay(0.0)
	_ Hook = hook.ClipGrad(0.0)
)

// Model is the interface implemented by optimizable models.
type Model interface {
	Params() layer.Parameters
}

// Hook transforms parameters before an optimizer update is applied.
type Hook func(params []layer.Parameter)

// Params returns model parameters that currently have gradients and applies hooks to them.
func Params(m Model, hook []Hook) []layer.Parameter {
	params := make([]layer.Parameter, 0)
	for _, p := range m.Params() {
		if p.Grad == nil {
			continue
		}

		params = append(params, p)
	}

	for _, h := range hook {
		h(params)
	}

	return params
}
