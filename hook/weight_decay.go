package hook

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/matrix"
)

func WeightDecay(lambda float64) func(params []layer.Parameter) {
	return func(params []layer.Parameter) {
		for _, p := range params {
			p.Data = matrix.F2(p.Data, p.Grad.Data, decay(lambda))
		}
	}
}

func decay(lambda float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a + lambda*b }
}
