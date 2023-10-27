package optimizer

import (
	"fmt"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Model = (*model.MLP)(nil)
	_ Model = (*model.LSTM)(nil)
)

var (
	_ Hook = hook.WeightDecay(0.0)
)

type Model interface {
	Params() layer.Parameters
}

type Hook func(params []layer.Parameter)

func Params(m Model) []layer.Parameter {
	params := make([]layer.Parameter, 0)
	for _, p := range m.Params() {
		if p.Grad == nil {
			continue
		}

		params = append(params, p)
	}

	return params
}

func id(p *variable.Variable) string {
	return fmt.Sprintf("%p", p)
}
