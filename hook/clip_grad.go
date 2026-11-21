package hook

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/math"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// ClipGrad returns a hook that clips the global gradient norm to max.
func ClipGrad(max float32) func(params layer.Parameters) {
	return func(params layer.Parameters) {
		var total float32
		for _, p := range params {
			if p.Grad == nil {
				continue
			}

			total += tensor.Sum(tensor.Pow(2, p.Grad.Data)).At()
		}

		rate := max / (math.Sqrt(total) + 1e-6)
		if rate >= 1 {
			return
		}

		// Shared gradients contribute per parameter to the norm, but are scaled once.
		scaled := make(map[*variable.Variable]bool)
		for _, p := range params {
			if p.Grad == nil || scaled[p.Grad] {
				continue
			}

			p.Grad.Data = tensor.MulC(rate, p.Grad.Data)
			scaled[p.Grad] = true
		}
	}
}
