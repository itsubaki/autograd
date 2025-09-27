package hook

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
)

func WeightDecay(lambda float64) func(params []layer.Parameter) {
	return func(params []layer.Parameter) {
		for _, p := range params {
			p.Grad.Data = tensor.F2(p.Grad.Data, p.Data, decay(lambda))
		}
	}
}

func decay(lambda float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a + lambda*b }
}
