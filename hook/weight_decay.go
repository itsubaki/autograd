package hook

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
)

// WeightDecay returns a hook that adds L2 weight decay to gradients.
func WeightDecay(lambda float32) func(params layer.Parameters) {
	return func(params layer.Parameters) {
		for _, p := range params {
			if p.Grad == nil {
				continue
			}

			p.Grad.Data = tensor.F2(p.Grad.Data, p.Data, decay(lambda))
		}
	}
}

// decay returns a function that adds lambda times b to a.
func decay(lambda float32) func(a, b float32) float32 {
	return func(a, b float32) float32 { return a + lambda*b }
}
