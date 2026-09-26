package hook

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/math"
	"github.com/itsubaki/autograd/tensor"
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

		for _, p := range params {
			if p.Grad == nil {
				continue
			}

			p.Grad.Data = tensor.MulC(rate, p.Grad.Data)
		}
	}
}
